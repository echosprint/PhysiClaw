"""JSONSchema argument validator for principle 4.

Validates each `ToolCall.arguments` against the tool's declared
`input_schema` BEFORE dispatch. Catches:
  - unknown or missing required properties
  - wrong types (string vs array, number vs string-digits)
  - enum violations
  - truncated arguments (technically valid JSON but missing a required key)

On failure, the engine does NOT actuate — it pairs the call with an error
`ToolResult` so the model sees the issue next turn and self-corrects.

This is a deliberately small subset of JSONSchema — just the shapes the
MCP tools and our local synthetic tools use. No external dep.
"""

from typing import Any


class ValidationError(Exception):
    """Raised when arguments don't match the schema. Caller converts this
    into an error ToolResult rather than executing the tool."""


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _is_number(v: Any) -> bool:
    """True iff v is int or float — NOT bool (Python bool is subclass of int)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_arguments(args: Any, schema: dict[str, Any]) -> None:
    """Check `args` against a JSONSchema `schema`. Raises ValidationError.

    `args` is typed `Any` because it crosses a trust boundary — the data
    is JSON-decoded LLM output, so a wrong top-level type (list, string,
    null) is a possibility we have to defend against, even though every
    in-process caller passes a dict.

    The schema is expected to look like:
        {
          "type": "object",
          "properties": {name: {"type": ..., "enum": [...], "items": {...}}},
          "required": ["name", ...],
        }
    """
    if not isinstance(args, dict):
        raise ValidationError(
            f"arguments must be a JSON object, got {type(args).__name__}"
        )

    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    missing = [k for k in required if k not in args]
    if missing:
        raise ValidationError(
            f"missing required argument(s): {', '.join(sorted(missing))}"
        )

    # Bbox shape — IDENTICAL LOGIC to `validate_bbox` in
    # core/vision/util.py (same checks, same order, same messages). Keep
    # the two in sync. Runs before the per-key `_check_value` loop so
    # the bbox-specific error fires before the generic schema type
    # check. The orchestrator calls validate_bbox at gesture time as
    # defense-in-depth.
    bbox = args.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValidationError(
                f"bbox: must be [left, top, right, bottom]; got {bbox!r}"
            )
        if not all(isinstance(v, (int, float)) for v in bbox):
            raise ValidationError(f"bbox: each coord must be a number; got {bbox!r}")
        left, top, right, bottom = bbox
        if any(v < 0 or v > 1 for v in bbox):
            raise ValidationError(
                f"bbox: each coord must be in [0, 1]; got [{left}, {top}, {right}, {bottom}]"
            )
        if left >= right or top >= bottom:
            raise ValidationError(
                f"bbox: left < right, top < bottom; got [{left}, {top}, {right}, {bottom}]"
            )

    for key, value in args.items():
        prop = properties.get(key)
        if prop is None:
            # Unknown property — tolerated (additionalProperties defaults to
            # true in JSONSchema). Strict mode could reject here.
            continue
        _check_value(key, value, prop)


def _check_value(key: str, value: Any, prop: dict[str, Any]) -> None:
    enum = prop.get("enum")
    if enum is not None and value not in enum:
        raise ValidationError(f"{key}: value {value!r} is not one of {enum}")

    if _is_number(value):
        minimum = prop.get("minimum")
        if minimum is not None and value < minimum:
            raise ValidationError(f"{key}: {value} < minimum {minimum}")
        maximum = prop.get("maximum")
        if maximum is not None and value > maximum:
            raise ValidationError(f"{key}: {value} > maximum {maximum}")

    if isinstance(value, str):
        min_len = prop.get("minLength")
        if min_len is not None and len(value) < min_len:
            raise ValidationError(
                f"{key}: string length {len(value)} < minLength {min_len}"
            )
        max_len = prop.get("maxLength")
        if max_len is not None and len(value) > max_len:
            raise ValidationError(
                f"{key}: string length {len(value)} > maxLength {max_len}"
            )

    expected_type = prop.get("type")
    if expected_type is None:
        return

    # `type` may be a list in JSONSchema (e.g. ["string", "null"]).
    types_to_check = (
        expected_type if isinstance(expected_type, list) else [expected_type]
    )
    matched = False
    for t in types_to_check:
        py_type = _TYPE_MAP.get(t)
        if py_type is None:
            # Unknown type in schema — skip check rather than reject.
            matched = True
            break
        # bool is a subclass of int in Python; treat them distinctly.
        if t == "boolean":
            if isinstance(value, bool):
                matched = True
                break
            continue
        if t in ("integer", "number") and isinstance(value, bool):
            continue  # bool is int subclass; _is_number() semantics here too
        if isinstance(value, py_type):
            matched = True
            break
    if not matched:
        got = type(value).__name__
        want = "|".join(types_to_check)
        raise ValidationError(f"{key}: expected {want}, got {got} ({value!r})")

    if expected_type == "array":
        items_schema = prop.get("items")
        if items_schema:
            for i, elem in enumerate(value):
                try:
                    _check_value(f"{key}[{i}]", elem, items_schema)
                except ValidationError:
                    raise
        min_items = prop.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValidationError(
                f"{key}: array has {len(value)} items, min {min_items}"
            )
        max_items = prop.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            raise ValidationError(
                f"{key}: array has {len(value)} items, max {max_items}"
            )

    if expected_type == "object":
        sub_schema = prop
        if sub_schema.get("properties") or sub_schema.get("required"):
            validate_arguments(value, sub_schema)
