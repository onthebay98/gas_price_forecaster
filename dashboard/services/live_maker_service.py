"""Service for managing the live_maker.py process from the dashboard."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from dashboard.utils.constants import (
    LIVE_MAKER_LOG,
    LIVE_MAKER_PID,
    PROJECT_ROOT,
)

# Metadata file stores launch options alongside the PID
_LIVE_MAKER_META = LIVE_MAKER_PID.with_suffix(".json")

# Tag -> HTML color mapping
TAG_COLORS = {
    "FILL": "#22c55e",
    "DIFF": "#4ade80",
    "DIFF DRY": "#4ade80",
    "PRUNE": "#eab308",
    "PRUNE DRY": "#eab308",
    "CYCLE": "#06b6d4",
    "STATE": "#eab308",
    "DEFENSIVE": "#ef4444",
    "TAKER": "#a855f7",
    "EXIT": "#3b82f6",
    "INIT": "#06b6d4",
    "LOOP": "#06b6d4",
    "PREFLIGHT": "#06b6d4",
    "SHUTDOWN": "#eab308",
    "DISCORD": "#a855f7",
    "TOXICITY": "#eab308",
    "SESSION SUMMARY": "#06b6d4",
    "HALT": "#ef4444",
    "FATAL": "#ef4444",
}


def is_running() -> bool:
    """Check if live_maker is running via PID file."""
    if not LIVE_MAKER_PID.exists():
        return False
    try:
        pid = int(LIVE_MAKER_PID.read_text().strip())
        os.kill(pid, 0)  # Signal 0 = check if alive
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file - clean up
        LIVE_MAKER_PID.unlink(missing_ok=True)
        return False


def get_pid() -> int | None:
    """Get the PID of the running live_maker process, or None."""
    if not LIVE_MAKER_PID.exists():
        return None
    try:
        pid = int(LIVE_MAKER_PID.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        LIVE_MAKER_PID.unlink(missing_ok=True)
        return None


def start(
    dry_run: bool = False,
    use_demo: bool = False,
    cancel_on_exit: bool = False,
) -> bool:
    """Launch live_maker.py as a detached subprocess. Returns True on success."""
    if is_running():
        return False  # Already running

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "live_maker.py"),
        "--log-file", str(LIVE_MAKER_LOG),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if use_demo:
        cmd.append("--use-demo")
    if cancel_on_exit:
        cmd.append("--cancel-on-exit")

    # Clear old log
    LIVE_MAKER_LOG.unlink(missing_ok=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # Detach from parent
    )
    # PID file is written by live_maker.py itself (_acquire_pid_lock).
    # Wait briefly for it to appear so the dashboard sees the status immediately.
    for _ in range(20):
        if LIVE_MAKER_PID.exists():
            break
        import time; time.sleep(0.1)
    _LIVE_MAKER_META.write_text(json.dumps({
        "dry_run": dry_run,
        "use_demo": use_demo,
        "cancel_on_exit": cancel_on_exit,
    }))
    return True


def stop() -> bool:
    """Send SIGTERM to the live_maker process. Returns True if signal sent.

    PID file cleanup is handled by the process itself (atexit handler).
    """
    pid = get_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        _LIVE_MAKER_META.unlink(missing_ok=True)
        # Wait briefly for process to exit and clean up its PID file
        for _ in range(30):
            try:
                os.kill(pid, 0)
                import time; time.sleep(0.1)
            except ProcessLookupError:
                break
        # If process didn't clean up, remove stale PID file
        if LIVE_MAKER_PID.exists():
            try:
                stale = int(LIVE_MAKER_PID.read_text().strip())
                os.kill(stale, 0)
            except (ValueError, ProcessLookupError, PermissionError):
                LIVE_MAKER_PID.unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        LIVE_MAKER_PID.unlink(missing_ok=True)
        _LIVE_MAKER_META.unlink(missing_ok=True)
        return False


def get_launch_options() -> dict | None:
    """Read the launch options of the currently running process, or None."""
    if not _LIVE_MAKER_META.exists():
        return None
    try:
        return json.loads(_LIVE_MAKER_META.read_text())
    except Exception:
        return None


def read_log(n_lines: int = 100) -> list[str]:
    """Read the last n_lines from the log file."""
    if not LIVE_MAKER_LOG.exists():
        return []
    try:
        lines = LIVE_MAKER_LOG.read_text().splitlines()
        return lines[-n_lines:]
    except Exception:
        return []


def format_log_html(lines: list[str]) -> str:
    """Convert log lines to colored HTML using [TAG] prefix matching."""
    if not lines:
        return '<span style="color: #888;">No log output yet.</span>'

    html_lines = []
    for line in lines:
        color = None

        # Check for ERROR/WARNING level
        if " [ERROR] " in line or " [CRITICAL] " in line:
            color = "#ef4444"
        elif " [WARNING] " in line:
            color = "#eab308"
        else:
            # Match [TAG] in the message part
            for tag, tag_color in TAG_COLORS.items():
                if f"[{tag}]" in line:
                    color = tag_color
                    break

        escaped = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if color:
            html_lines.append(f'<span style="color: {color};">{escaped}</span>')
        else:
            html_lines.append(f'<span style="color: #d4d4d4;">{escaped}</span>')

    return "\n".join(html_lines)
