# CLAUDE.md

@AGENTS.md

AGENTS.md is the single source of truth for this repo; edit it there, not here. Claude-specific notes:

- Hardware scripts block on `input()`, keyboard listeners (pynput) or Rerun windows; never start them from
  a Bash tool call, not even in the background. Hand the exact command to the operator instead.
- Verify changes with `uv run --frozen pytest test/` (CPU, no hardware) before reporting them as done.
