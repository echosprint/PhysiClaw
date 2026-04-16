---
name: setup
description: Connect the robotic arm and camera, then calibrate. Required before using any PhysiClaw MCP tools.
allowed-tools: Bash
---

# Setup

```bash
uv run python scripts/setup.py
```

Fails with non-zero exit and prints which step failed. Fix the physical setup and rerun.
