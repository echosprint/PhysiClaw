"""`python -m physiclaw.cli` — the console script's module twin, so a
child process can be spawned the way `cli/server.py` spawns the runtime
(`sys.executable -m …`) without depending on the script being on PATH."""

from physiclaw.cli import app

app()
