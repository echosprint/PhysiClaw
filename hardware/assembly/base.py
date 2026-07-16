"""Base class for assembly-step drawings.

``BaseAssembly`` inherits from ``BasePart`` — assemblies share the
build/export/output_path machinery (so ``.export()`` writes a STEP of
the composed assembly, handy for CAD inspection) and add ``.render()``
for the SVG drawing used in the manual.

Default filenames:
  * ``hardware/output/step/<module_name>_<variant>.step`` (inherited)
  * ``hardware/output/svg/<module_name>_<variant>_cam<i>.svg`` (this class)
"""

import warnings
from pathlib import Path

from build123d import MM, Compound, ExportSVG, LineType, ShapeList, Unit

from hardware.assembly.projection import ISO, Camera, camera_view
from hardware.assembly.svg_utils import inject_non_scaling_strokes, strip_root_dims
from hardware.parts.base import BasePart
from hardware.scheme import svg_path_for, variant_suffix

# Top-level Compound labels recognised by BaseAssembly.render() for the
# two-layer split. The ``layer_`` prefix namespaces them as render-routing
# tags so they can't collide with a normal part label like "solid" or
# "ghost" that a procedure might choose for an unrelated reason.
SOLID_LABEL = "_layer_solid"
GHOST_LABEL = "_layer_ghost"


# View tags for the ``views`` declaration — (variant, camera index) pairs
# named after the SVG they produce, so a declaration reads like the output
# filenames: ``views = [EXPLODED_CAM1, ASSEMBLED_CAM0]`` renders
# ``<stem>_exploded_cam1.svg`` and ``<stem>_assembled_cam0.svg``. A stem
# with a third camera can tag it inline as ``("exploded", 2)``.
EXPLODED_CAM0 = ("exploded", 0)
EXPLODED_CAM1 = ("exploded", 1)
ASSEMBLED_CAM0 = ("assembled", 0)
ASSEMBLED_CAM1 = ("assembled", 1)


class BaseAssembly(BasePart):
    """Buildable assembly — STEP via .export() (inherited), SVG via .render().

    Two-layer SVG: if ``_build`` returns a Compound with top-level children
    labeled ``SOLID_LABEL`` and ``GHOST_LABEL``, they are projected to
    separate SVG layers (ghost = lighter + phantom dashes) — exploded-view
    illustration of prep state vs result. Otherwise the whole assembly is
    one layer.

    Every assembly has two variants: ``exploded=True`` shows the install
    motion (gaps + ghosts), ``exploded=False`` shows the finished state.
    The flag is exposed as a ctor kwarg so callers can ``export()`` both
    from one ``__main__``, and so a downstream assembly can embed an
    upstream one in its assembled form (e.g. ``FR20SHCS(exploded=False)``).
    Output filenames are suffixed ``_exploded`` / ``_assembled`` to keep
    both on disk side by side, plus ``_cam<i>`` (always present, ``cam0``
    for single-camera assemblies) so the scheme is uniform whether
    ``camera`` is a single Camera or a list.
    """

    # One camera → one SVG per variant; a list of cameras → one SVG per
    # camera per variant, filenames suffixed ``_cam0``, ``_cam1``, ….
    camera: "Camera | list[Camera]" = ISO
    # Which views actually get rendered — a list of the view tags above,
    # e.g. ``views = [EXPLODED_CAM1, ASSEMBLED_CAM0]``. None (default)
    # renders every camera for both variants; a variant with no tag in the
    # list renders no SVG at all (its STEP still exports — kept for CAD
    # inspection). Declare only the views a consumer (the manual) needs:
    # every skipped view saves an exact-HLR pass, which is both the
    # dominant render cost and the crash surface. Indices refer to
    # positions in ``cameras`` and stay stable in filenames, so manual
    # references and patch JSONs never rename when views are trimmed.
    views: "list[tuple[str, int]] | None" = None
    # With `vector-effect: non-scaling-stroke` baked into every render,
    # these values are interpreted as ~device pixels (not millimetres) —
    # sub-pixel widths still render via anti-aliasing. 0.8 / 0.4 chosen
    # by eye: heavy enough to read at full zoom, light enough not to
    # crowd the drawing.
    line_weight: float = 0.8
    page_margin: float = 5 * MM
    ghost_line_weight: float = 0.4
    ghost_line_type: LineType = LineType.PHANTOM

    def __init__(self, *, exploded: bool = False):
        super().__init__()
        self.exploded = exploded

    def name_suffix(self) -> str:
        # Assemblies are one-offs; drop the inherited "_x{qty}" suffix and
        # use the variant tag instead, so the STEP filename is e.g.
        # solenoid_tip_exploded.step / solenoid_tip_assembled.step.
        return variant_suffix(self.exploded)

    def bom_key(self):
        return None  # assemblies are structural; their parts register themselves

    def svg_path(self, index: int | None = None) -> Path:
        return svg_path_for(self._module_stem(), self.exploded, index=index)

    # Assemblies deliberately DO NOT opt into the geometry cache
    # (``geom_key`` stays the BasePart default of ``None``). The cache
    # returns ``copy.copy()`` of the cached shape, and build123d's
    # ``copy.copy`` is a full ``copy.deepcopy`` — every face is
    # ``BRepBuilderAPI_Copy``'d and the anytree children recursed. For a
    # leaf part that's one cheap copy of an expensive-to-build solid, a
    # clear win. For an assembly *compound* it deep-duplicates the entire
    # accumulated tree, so caching it costs O(all faces) per reuse and
    # the cost compounds up the chain — measured at ~50% of build time
    # and a ~10× peak-memory blow-up on the deepest procedure. An
    # assembly is cheap to recompose from its already-cached leaf parts,
    # so we just rebuild it instead of caching+deep-copying.

    def _build(self) -> Compound:
        raise NotImplementedError

    @property
    def cameras(self) -> "list[Camera]":
        """``camera`` normalized to a list; ``_cam<i>`` filenames index
        into it. Which entries actually render is ``view_indices``."""
        return self.camera if isinstance(self.camera, list) else [self.camera]

    @property
    def view_indices(self) -> "list[int]":
        """Camera indices rendered for THIS variant (``self.exploded``) —
        the single source of truth shared by ``render()`` (producer) and
        the build dispatcher's completeness check (verifier) so they
        cannot drift. ``views`` unset → every camera. Validates the
        declaration so a typo'd variant name or out-of-range index fails
        the build instead of silently rendering nothing."""
        n = len(self.cameras)
        if self.views is None:
            return list(range(n))
        name = type(self).__name__
        if unknown := {v for v, _ in self.views} - {"exploded", "assembled"}:
            raise ValueError(
                f"{name}.views has unknown variant(s) {sorted(unknown)}; "
                f"use the EXPLODED_CAM* / ASSEMBLED_CAM* tags"
            )
        if len(set(self.views)) != len(self.views):
            raise ValueError(f"{name}.views has duplicate view tags")
        variant = "exploded" if self.exploded else "assembled"
        indices = [i for v, i in self.views if v == variant]
        if bad := [i for i in indices if not 0 <= i < n]:
            raise ValueError(
                f"{name}.views: camera indices {bad} out of range for {n} camera(s)"
            )
        return indices

    def render(self, *, only_missing: bool = False) -> None:
        """Project every declared view (``view_indices``) to its SVG. With
        ``only_missing``, skip cameras whose SVG already exists — the
        crash-retry path uses this
        so a re-run never repeats an HLR projection that already landed
        (each exact-HLR pass is another roll of the OCCT SIGSEGV dice).
        Callers own the staleness guarantee: an existing SVG must be from
        the current build cycle (the build dispatcher ensures this by
        clearing a stem's outputs before its first attempt) — and the
        whole-variant skip that avoids ``build()`` entirely lives with
        them too, since only they can also skip the STEP export."""
        if not self.view_indices:  # variant renders no views — skip the build
            return
        assembly = self.build()
        solid, ghost = _split_solid_ghost(assembly)

        with warnings.catch_warnings():
            # OCCT's hidden-line removal can split a projected circular rim (every
            # pulley/idler flange/bore) at its silhouette-tangent point and leave a
            # sub-micron degenerate ellipse; ExportSVG then warns it is "too small
            # to export safely". The skipped remnant is ~1e-6 mm — invisible — so
            # the warning is pure noise. Silence just that one message.
            warnings.filterwarnings(
                "ignore", message="Skipping ellipse that is too small"
            )
            for i in self.view_indices:
                cam = self.cameras[i]
                if only_missing and self.svg_path(index=i).exists():
                    continue
                # Camera + look_at derived from the FULL assembly bbox so
                # solid and ghost layers align pixel-for-pixel. Without a
                # shared look_at, project_to_viewport defaults to each
                # subset's own center, which warps the projection direction
                # per layer.
                cam_pos, up, look_at = camera_view(assembly, cam)

                exporter = ExportSVG(unit=Unit.MM, margin=self.page_margin)
                exporter.add_layer(SOLID_LABEL, line_weight=self.line_weight)
                if ghost is not None:
                    exporter.add_layer(
                        GHOST_LABEL,
                        line_weight=self.ghost_line_weight,
                        line_type=self.ghost_line_type,
                    )

                solid_visible, _ = solid.project_to_viewport(
                    cam_pos, up, look_at=look_at
                )
                exporter.add_shape(ShapeList(solid_visible), layer=SOLID_LABEL)
                if ghost is not None:
                    ghost_visible, _ = ghost.project_to_viewport(
                        cam_pos, up, look_at=look_at
                    )
                    exporter.add_shape(ShapeList(ghost_visible), layer=GHOST_LABEL)

                # Write via a temp name and rename into place so the final
                # path only ever holds a fully post-processed SVG — the
                # crash-retry's only_missing skip trusts any existing file.
                path = self.svg_path(index=i)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                exporter.write(str(tmp))
                tmp.write_text(
                    inject_non_scaling_strokes(strip_root_dims(tmp.read_text()))
                )
                tmp.replace(path)


def _split_solid_ghost(assembly):
    """Return (solid, ghost) Compounds. Looks for top-level children
    labeled SOLID_LABEL / GHOST_LABEL; if no SOLID_LABEL child is
    present, the whole assembly is treated as solid so existing
    single-layer assemblies render unchanged.

    Uses ``dict.get`` (not a truthiness fallback) so an empty Compound
    is still recognised as the solid layer rather than slipping through
    to the assembly-root fallback — empty Compounds are falsy in
    build123d but a labeled-but-empty solid layer is still meaningful.

    Raises ValueError if either label appears more than once — last-wins
    silent overwrite is a footgun; merge the shapes under one Compound
    explicitly instead."""
    found: dict[str, Compound] = {}
    for child in getattr(assembly, "children", []):
        label = getattr(child, "label", None)
        if label not in (SOLID_LABEL, GHOST_LABEL):
            continue
        if label in found:
            raise ValueError(
                f"duplicate top-level {label!r} child in assembly "
                f"{assembly.label!r}; merge into one Compound"
            )
        found[label] = child
    solid = found.get(SOLID_LABEL, assembly)
    return solid, found.get(GHOST_LABEL)
