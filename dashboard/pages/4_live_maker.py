"""
Live Maker Page

Start/stop the live maker loop and view its log output.
"""
import sys
import time
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from dashboard.services.live_maker_service import (
    format_log_html,
    get_launch_options,
    get_pid,
    is_running,
    read_log,
    start,
    stop,
)

st.set_page_config(
    page_title="Live Maker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Live Maker")

# --- Status ---
running = is_running()
pid = get_pid()
opts = get_launch_options() if running else None

if running:
    labels = []
    if opts:
        if opts.get("dry_run"):
            labels.append("dry-run")
        if opts.get("use_demo"):
            labels.append("demo")
        if opts.get("cancel_on_exit"):
            labels.append("cancel-on-exit")
    mode_str = f" ({', '.join(labels)})" if labels else ""
    st.markdown(f':green-background[Running] &nbsp; PID: `{pid}`{mode_str}')
else:
    st.markdown(':red-background[Stopped]')

# --- Controls ---
st.subheader("Controls")

col_opts, col_actions = st.columns([2, 1])

with col_opts:
    c1, c2, c3 = st.columns(3)
    dry_run = c1.checkbox("Dry run", value=False, disabled=running)
    use_demo = c2.checkbox("Demo API", value=False, disabled=running)
    cancel_on_exit = c3.checkbox(
        "Cancel orders on stop", value=False, disabled=running,
        help="Cancel all resting orders when the process is stopped. Leave unchecked for restarts.",
    )

with col_actions:
    c_start, c_stop = st.columns(2)
    with c_start:
        if st.button(
            "Start",
            type="primary",
            disabled=running,
            use_container_width=True,
        ):
            ok = start(dry_run=dry_run, use_demo=use_demo, cancel_on_exit=cancel_on_exit)
            if ok:
                st.success("Started")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Failed to start (already running?)")
    with c_stop:
        if st.button(
            "Stop",
            type="secondary",
            disabled=not running,
            use_container_width=True,
        ):
            ok = stop()
            if ok:
                st.warning("Stop signal sent")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Failed to stop (not running?)")

# --- Log viewer ---
st.subheader("Log Output")

log_lines = read_log(n_lines=200)
html = format_log_html(log_lines)

components.html(
    f"""<div id="live-maker-log" style="
        background-color: #1e1e1e;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 12px;
        line-height: 1.5;
        height: 600px;
        overflow-y: auto;
        white-space: pre;
    ">{html}</div>
    <script>
        const el = document.getElementById('live-maker-log');
        if (el) el.scrollTop = el.scrollHeight;
    </script>""",
    height=640,
    scrolling=False,
)

# Auto-refresh every 2 seconds while running
if running:
    time.sleep(2)
    st.rerun()
