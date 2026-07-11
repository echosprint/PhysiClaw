"""CLI entry point — opens the browser on the served SVG.

    uv run --group cad python -m hardware.assembly.mark <input.svg>
    uv run --group cad python -m hardware.assembly.mark <patch.json>

Passing a ``patch/<stem>.json`` resolves its source SVG in output/svg and
opens it with the existing patch loaded for editing (select / move / recolour).
"""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path
from typing import List

from hardware.assembly.mark.patch import source_for_patch
from hardware.assembly.mark.server import make_server


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m hardware.assembly.mark <input.svg | patch.json>",
            file=sys.stderr,
        )
        return 2
    arg = Path(argv[1]).expanduser().resolve()
    if not arg.exists():
        print(f"file not found: {arg}", file=sys.stderr)
        return 2
    if arg.suffix.lower() == ".json":
        # A patch file → edit its source SVG with the patch loaded.
        src = source_for_patch(arg)
        if not src.exists():
            print(
                f"no source SVG for patch {arg.name}: expected {src}", file=sys.stderr
            )
            return 2
    elif arg.suffix.lower() == ".svg":
        src = arg
    else:
        print(f"not an .svg or .json: {arg}", file=sys.stderr)
        return 2

    server = make_server(src)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"mark: serving {src.name}")
    print(f"      open {url}  (Ctrl+C to quit)")
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
