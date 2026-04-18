---
name: setup
description: Connect the robotic arm and camera, then calibrate. Required before using any PhysiClaw MCP tools.
allowed-tools: Bash
---

# Setup

```bash
uv run python scripts/setup.py           # interactive, default
uv run python scripts/setup.py -y        # auto mode, skip prompts
uv run python scripts/setup.py --trace   # add edge-trace visual check at end
```

Fails with non-zero exit and prints which step failed. Fix the physical setup and rerun.
