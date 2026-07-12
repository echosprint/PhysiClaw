"""``physiclaw setup phone`` — learn the on-screen keyboard.

Takes phone screenshots with the keyboard visible, detects key positions,
and writes a UI-preset markdown file that the agent uses for typing.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from physiclaw.common import paths
from physiclaw.common.text import write_text


def phone(
    images: Annotated[
        list[Path],
        typer.Argument(
            help="Phone screenshots with keyboard visible (alpha + numeric).",
            exists=True,
            readable=True,
        ),
    ],
    bbox_dir: Annotated[
        Path,
        typer.Option("--bbox-dir", help="Directory for bounding-box preview images."),
    ] = Path("keyboard-bbox"),
    preset: Annotated[
        Path,
        typer.Option(
            "--preset",
            help="Output preset file. Defaults to "
            "~/.physiclaw/ui-presets/system-keyboard.md.",
        ),
    ] = paths.ui_presets_dir() / "system-keyboard.md",
) -> None:
    """Detect keyboard keys from phone screenshots and generate a UI preset.

    Pass one or more screenshot paths (alpha + numeric layouts). The output
    preset file has a handful of key slots marked ??? — fill those in
    manually from the bbox previews.

    Example:
        physiclaw setup phone alpha.png numeric.png
    """
    import cv2

    from physiclaw.core.vision.keyboard import (
        detect_key_boxes,
        draw_detected_keys,
        generate_preset,
        label_keyboard,
    )

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    bbox_dir.mkdir(parents=True, exist_ok=True)

    pages: dict = {}
    bbox_images: dict = {}

    for img_path in images:
        typer.echo(f"\n{'=' * 60}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            typer.echo(f"Error: cannot read {img_path}")
            continue

        h, w = frame.shape[:2]
        typer.echo(f"Image: {img_path.name} ({w}x{h})")

        rows = label_keyboard(frame)
        if rows is None:
            typer.echo("No keys detected")
            continue

        row_counts = [len(row) for row in rows]
        total = sum(row_counts)
        is_numeric = (
            len(row_counts) >= 2 and row_counts[0] == 10 and row_counts[1] == 10
        )
        page_name = "Numeric Keyboard" if is_numeric else "Alpha Keyboard"
        typer.echo(f"Page: {page_name} ({total} keys, rows: {row_counts})")

        boxes, bg = detect_key_boxes(frame)
        bbox_img = draw_detected_keys(frame, boxes, bg)
        bbox_path = bbox_dir / f"bbox_{img_path.stem}.png"
        cv2.imwrite(str(bbox_path), bbox_img)
        typer.echo(f"Bounding box image: {bbox_path}")

        if page_name not in pages:
            pages[page_name] = rows
            bbox_images[page_name] = str(bbox_path)
        else:
            typer.echo(f"  (skipped — {page_name} already captured)")

        for i, row in enumerate(rows):
            labels = [k["element"] for k in row]
            typer.echo(f"  Row {i + 1}: {' '.join(labels)}")

    if not pages:
        typer.echo("No keyboard detected in any input image.")
        raise typer.Exit(code=1)

    rendered = generate_preset(pages, bbox_images)
    preset.parent.mkdir(parents=True, exist_ok=True)
    write_text(preset, rendered)
    ref_path = bbox_dir / "system-keyboard.ref.md"
    write_text(ref_path, rendered)
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Preset written to {preset}")
    typer.echo(f"Reference copy saved to {ref_path}")
    typer.echo("Keys marked ??? need to be filled in from the bbox previews.")
