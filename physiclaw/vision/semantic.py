"""Convert UI element JSON to LLM-friendly semantic description.

Pipeline: elements JSON → zone clustering → spatial enrichment
        → semantic text (for LLM) + id-to-bbox lookup (for execution)
"""

from dataclasses import dataclass, field


# ── Data model ─────────────────────────────────


@dataclass
class UIElement:
    id: int
    elem_type: str      # button | text | icon | input | image
    label: str           # display text, may be empty
    bbox: list[float]   # [left, top, right, bottom] normalized 0-1
    conf: float = 1.0
    zone: str = ""

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Zone:
    name: str
    y_min: float
    y_max: float
    elements: list[UIElement] = field(default_factory=list)


# ── Parse ──────────────────────────────────────

_TYPE_MAP = {
    "button": "button", "btn": "button",
    "text": "text", "icon": "icon",
    "input": "input", "textbox": "input",
    "image": "image", "img": "image",
}


def parse_elements(raw: list[dict]) -> list[UIElement]:
    """Parse ui_elements JSON into UIElement list."""
    elements = []
    for item in raw:
        label_str = item.get("label", "")
        if ":" in label_str:
            elem_type, label = label_str.split(":", 1)
            elem_type = elem_type.strip().lower()
            label = label.strip()
        else:
            elem_type = label_str.strip().lower()
            label = ""
        elem_type = _TYPE_MAP.get(elem_type, elem_type)
        elements.append(UIElement(
            id=item["id"],
            elem_type=elem_type,
            label=label,
            bbox=item["bbox"],
            conf=item.get("conf", 1.0),
        ))
    return elements


# ── Zone clustering ────────────────────────────

DEFAULT_ZONES = [
    Zone("status-bar",    0.00, 0.06),
    Zone("nav-bar",       0.06, 0.14),
    Zone("main-content",  0.14, 0.44),
    Zone("detail-area",   0.44, 0.60),
    Zone("list-area",     0.60, 0.88),
    Zone("bottom-bar",    0.88, 1.00),
]

_ZONE_NAMES = {0: "top", 1: "upper", 2: "middle", 3: "lower"}


def auto_cluster_zones(elements: list[UIElement],
                       gap_threshold: float = 0.04) -> list[Zone]:
    """Auto-detect zones by finding vertical gaps between elements."""
    if not elements:
        return []
    sorted_els = sorted(elements, key=lambda e: e.cy)
    splits = [0.0]
    for i in range(1, len(sorted_els)):
        gap = sorted_els[i].bbox[1] - sorted_els[i - 1].bbox[3]
        if gap > gap_threshold:
            splits.append((sorted_els[i - 1].bbox[3] + sorted_els[i].bbox[1]) / 2)
    splits.append(1.0)

    zones = []
    total = len(splits) - 1
    for i in range(total):
        if i == total - 1:
            name = "bottom"
        elif total <= 4:
            name = _ZONE_NAMES.get(i, f"zone-{i+1}")
        else:
            name = f"zone-{i+1}"
        zones.append(Zone(name, splits[i], splits[i + 1]))
    return zones


def assign_zones(elements: list[UIElement],
                 zones: list[Zone]) -> list[Zone]:
    """Assign each element to its zone based on vertical center."""
    for z in zones:
        z.elements = []
    for el in elements:
        assigned = False
        for z in zones:
            if z.y_min <= el.cy < z.y_max:
                el.zone = z.name
                z.elements.append(el)
                assigned = True
                break
        if not assigned:
            nearest = min(zones, key=lambda z: abs(el.cy - (z.y_min + z.y_max) / 2))
            el.zone = nearest.name
            nearest.elements.append(el)
    for z in zones:
        z.elements.sort(key=lambda e: (round(e.cy, 2), e.cx))
    return zones


# ── Spatial description ────────────────────────

_SIZE_BRACKETS = [
    (0.15, "xs"), (0.35, "sm"), (0.65, ""),
    (0.85, "lg"), (1.01, "xl"),
]


def _position(el: UIElement) -> str:
    if el.cx < 0.3:
        return "left"
    if el.cx > 0.7:
        return "right"
    return "full-width" if el.width > 0.7 else "center"


def _size_labels(elements: list[UIElement]) -> dict[int, str]:
    """Assign relative size labels (xs/sm/md/lg/xl) by area percentile."""
    if not elements:
        return {}
    areas = sorted(
        [(el.id, el.width * el.height) for el in elements],
        key=lambda x: x[1],
    )
    n = max(len(areas) - 1, 1)
    labels = {}
    for rank, (eid, _) in enumerate(areas):
        pct = rank / n
        for threshold, tag in _SIZE_BRACKETS:
            if pct < threshold:
                labels[eid] = tag
                break
    return labels


def _relationships(elements: list[UIElement]) -> dict[int, list[str]]:
    """Find left/right neighbor relationships."""
    rels: dict[int, list[str]] = {el.id: [] for el in elements}
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if i >= j:
                continue
            v_overlap = a.bbox[1] < b.bbox[3] and a.bbox[3] > b.bbox[1]
            if v_overlap and abs(a.cy - b.cy) < 0.02 and a.cx < b.cx:
                rels[a.id].append(f"left-of [{b.id}]")
                rels[b.id].append(f"right-of [{a.id}]")
    return rels


# ── Filtering ──────────────────────────────────


def _filter(elements: list[UIElement],
            task_hint: str | None = None,
            conf_threshold: float = 0.15) -> list[UIElement]:
    """Filter low-confidence elements, boost task-relevant ones."""
    filtered = [e for e in elements if e.conf >= conf_threshold]
    if task_hint:
        keywords = [kw for kw in task_hint.split() if len(kw) > 1]
        for el in filtered:
            if any(kw in el.label for kw in keywords):
                el.conf = max(el.conf, 0.95)
    return filtered


# ── Output formatters ──────────────────────────


def format_for_llm(zones: list[Zone],
                   relationships: dict[int, list[str]],
                   size_labels: dict[int, str] | None = None,
                   show_conf: bool = False) -> str:
    """Generate semantic text for LLM consumption.

    Example:
    ── bottom-bar
       [28] button "Buy Now" · xl · full-width
    ── list-area
       [24] button "Product A $29.9" · lg · full-width
    """
    lines = []
    for z in zones:
        if not z.elements:
            continue
        lines.append(f"── {z.name}")
        for el in z.elements:
            label_part = f' "{el.label}"' if el.label else ""
            line = f"   [{el.id}] {el.elem_type}{label_part}"

            if size_labels:
                tag = size_labels.get(el.id, "")
                if tag:
                    line += f" · {tag}"

            line += f" · {_position(el)}"

            rels = relationships.get(el.id, [])
            if rels:
                line += f" ({', '.join(rels)})"

            if show_conf and el.conf < 0.5:
                line += f" [low-conf:{el.conf:.0%}]"

            lines.append(line)
    return "\n".join(lines)


def build_action_lookup(elements: list[UIElement]) -> dict[int, dict]:
    """Build id->bbox lookup for the execution layer."""
    return {
        el.id: {
            "bbox": el.bbox,
            "center": [round(el.cx, 4), round(el.cy, 4)],
            "type": el.elem_type,
            "label": el.label,
        }
        for el in elements
    }


# ── Main entry point ──────────────────────────


class SemanticDescriber:
    """Convert UI element JSON to semantic text + action lookup.

    Usage:
        describer = SemanticDescriber()
        result = describer.describe(raw_elements, task_hint="buy watermelon")
        result["prompt"]   # feed to LLM
        result["lookup"]   # id->bbox for execution
    """

    def __init__(self,
                 zones: list[Zone] | None = None,
                 auto_zone: bool = False,
                 conf_threshold: float = 0.15,
                 show_conf: bool = True):
        self.zones = zones
        self.auto_zone = auto_zone
        self.conf_threshold = conf_threshold
        self.show_conf = show_conf

    def describe(self,
                 raw_elements: list[dict],
                 task_hint: str | None = None) -> dict:
        """Full pipeline: parse → filter → zone → spatial → format."""
        elements = parse_elements(raw_elements)
        elements = _filter(elements, task_hint, self.conf_threshold)

        if self.auto_zone or self.zones is None:
            zones = auto_cluster_zones(elements)
        else:
            zones = [Zone(z.name, z.y_min, z.y_max) for z in self.zones]
        zones = assign_zones(elements, zones)

        rels = _relationships(elements)
        sizes = _size_labels(elements)
        prompt = format_for_llm(zones, rels, sizes, self.show_conf)
        lookup = build_action_lookup(elements)

        if task_hint:
            prompt = f"task: {task_hint}\n\n{prompt}"

        return {
            "prompt": prompt,
            "lookup": lookup,
            "element_count": len(elements),
            "zones_used": [z.name for z in zones if z.elements],
        }
