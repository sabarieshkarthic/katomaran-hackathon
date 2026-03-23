"""
app.py — Streamlit Frontend
============================
Issue 1 fix : Tab switching was killing the pipeline because Streamlit reruns
              the entire script on every interaction (including tab click).
              Fix: store pipeline state in st.session_state. The video path,
              IdentityManager, FrameProcessor, and VideoCapture are kept in
              session_state across reruns. On each rerun (tab switch), the
              pipeline checks if it should keep running and continues from
              where it left off using st.rerun() to drive the loop.

Issue 1b fix: Each video gets its own per-video log folder via
              logger_utils.make_session_logs_dir(). The log tab shows only
              the current video's events.log, not all videos combined.

Issue 4 fix : Sidebar shows per-video unique count (from IdentityManager
              in-memory counter) separately from all-time DB count.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, tempfile, cv2, numpy as np
import streamlit as st
import pandas as pd

import database.db_manager as db_manager
import utils.logger as logger_utils
from identity_manager import IdentityManager
from frame_processor import FrameProcessor

# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Tracker", page_icon="👁", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .stApp{background:#0d0f14;color:#e0e0e0}
  section[data-testid="stSidebar"]{background:#111318!important}
  .mc{background:#1a1d24;border:1px solid #2a2d36;border-radius:8px;
      padding:14px 18px;margin-bottom:10px;text-align:center}
  .mv{font-size:2rem;font-weight:700;font-family:monospace;color:#00e676}
  .mvb{color:#40c4ff!important}.mva{color:#ffab40!important}.mvr{color:#ef5350!important}
  .ml{font-size:.78rem;color:#8a8fa8;letter-spacing:.08em;text-transform:uppercase;margin-top:2px}
  .log-box{background:#0a0c10;border:1px solid #1e2130;border-radius:6px;padding:12px;
           font-family:monospace;font-size:.75rem;color:#64e86e;max-height:340px;overflow-y:auto}
  h1,h2,h3{color:#e8eaf6!important}
</style>
""", unsafe_allow_html=True)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def init_system(cfg):
    db_manager.init_db(cfg["db_path"])
    logger_utils.ensure_base_logs_dir(cfg["logs_dir"])


# ─────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "processing":       False,   # True while video is running
        "video_path":       None,
        "session_logs_dir": None,    # per-video log folder
        "frame_idx":        0,
        "total_frames":     0,
        "idm":              None,
        "proc":             None,
        "cap":              None,
        "active_count":     0,
        "video_unique":     0,       # Issue 4: per-video unique count
        "video_entries":    0,
        "video_exits":      0,
        "done":             False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────
def start_pipeline(video_path: str, video_filename: str, cfg: dict):
    """Initialise all pipeline objects and store in session_state."""
    # Close any previous capture
    if st.session_state.get("cap") is not None:
        try: st.session_state["cap"].release()
        except: pass

    # Issue 1b: create per-video session log folder
    session_logs = logger_utils.make_session_logs_dir(cfg["logs_dir"], video_filename)

    idm  = IdentityManager(cfg, session_logs)
    proc = FrameProcessor(cfg, idm)
    cap  = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 99999

    st.session_state.update({
        "processing":       True,
        "video_path":       video_path,
        "session_logs_dir": session_logs,
        "frame_idx":        0,
        "total_frames":     total,
        "idm":              idm,
        "proc":             proc,
        "cap":              cap,
        "active_count":     0,
        "video_unique":     0,
        "video_entries":    0,
        "video_exits":      0,
        "done":             False,
    })


def stop_pipeline():
    """Flush remaining tracks and release capture."""
    idm = st.session_state.get("idm")
    cap = st.session_state.get("cap")
    if idm:
        try: idm.flush_all()
        except: pass
    if cap:
        try: cap.release()
        except: pass
    st.session_state["processing"] = False
    st.session_state["done"]       = True
    st.session_state["cap"]        = None


# ─────────────────────────────────────────────────────────────────────
def render_sidebar(cfg: dict):
    """
    Issue 4: show per-video unique count AND all-time DB count separately.
    """
    all_stats = db_manager.get_stats(cfg["db_path"])
    vid_unique  = st.session_state.get("video_unique", 0)
    vid_entries = st.session_state.get("video_entries", 0)
    vid_exits   = st.session_state.get("video_exits", 0)
    active      = st.session_state.get("active_count", 0)

    with st.sidebar:
        st.markdown("## 👁 Face Tracker")
        st.markdown("---")
        st.markdown("**This Video**")
        st.markdown(
            '<div class="mc"><div class="mv">{}</div><div class="ml">Unique (this video)</div></div>'
            '<div class="mc"><div class="mv mvb">{}</div><div class="ml">In Frame Now</div></div>'
            '<div class="mc"><div class="mv mva">{}</div><div class="ml">Entries (this video)</div></div>'
            '<div class="mc"><div class="mv mvr">{}</div><div class="ml">Exits (this video)</div></div>'
            .format(vid_unique, active, vid_entries, vid_exits),
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**All-Time DB**")
        st.markdown(
            '<div class="mc"><div class="mv" style="font-size:1.3rem">{}</div>'
            '<div class="ml">Total unique (all videos)</div></div>'
            .format(all_stats["unique_visitors"]),
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Config")
        st.markdown(
            "<div style='font-size:.8rem;color:#8a8fa8;font-family:monospace'>"
            "frame_skip:{fs} | face_thr:{ft} | body_thr:{bt}<br>"
            "exit_frames:{ef} | min_face:{mf}px | conf:{yc}<br>"
            "tracker: ByteTrack"
            "</div>".format(
                fs=cfg["frame_skip"], ft=cfg["face_threshold"],
                bt=cfg["body_threshold"], ef=cfg["exit_frames"],
                mf=cfg["min_face_size"], yc=cfg.get("model_confidence", 0.3),
            ),
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            "<div style='font-size:.8rem'>"
            "<span style='color:#00e676'>&#9632;</span> Face clear (Case A)<br>"
            "<span style='color:#40c4ff'>&#9632;</span> Body/face-unclear (Case B/C)<br>"
            "<span style='color:#aaa'>&#9632;</span> Skip frame"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────
def render_live_tab(cfg: dict):
    """
    Issue 1 fix: pipeline state lives in session_state, not inside
    the button callback. Tab switching triggers a rerun but
    session_state["processing"] is still True, so the loop continues.
    """
    # ── Upload area (only shown when not processing) ──────────────────
    if not st.session_state["processing"] and not st.session_state["done"]:
        uploaded = st.file_uploader(
            "Upload a video file", type=["mp4","avi","mov","mkv"],
        )
        if uploaded is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.flush()
            tfile.close()
            if st.button("▶ Start Processing", type="primary"):
                start_pipeline(tfile.name, uploaded.name, cfg)
                st.rerun()

    # ── Processing loop ───────────────────────────────────────────────
    if st.session_state["processing"]:
        cap   = st.session_state["cap"]
        idm   = st.session_state["idm"]
        proc  = st.session_state["proc"]
        total = st.session_state["total_frames"]

        # Stop button
        if st.button("⏹ Stop Processing"):
            stop_pipeline()
            st.rerun()

        frame_ph = st.empty()
        prog_ph  = st.progress(0.0)
        stat_ph  = st.empty()

        # Process a BATCH of frames per Streamlit rerun (not just one).
        # This prevents the UI from freezing on very high-fps videos
        # while still yielding to Streamlit between batches.
        BATCH = 5
        for _ in range(BATCH):
            ret, frame = cap.read()
            if not ret:
                stop_pipeline()
                st.rerun()
                break

            st.session_state["frame_idx"] += 1
            idx = st.session_state["frame_idx"]

            annotated, fstats = proc.process_frame(frame)

            # Update per-video counters from IDM
            st.session_state["active_count"]  = idm.get_active_count()
            st.session_state["video_unique"]  = idm.get_video_unique_count()

            frame_ph.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True,
            )
            pct = float(idx) / float(max(total, 1))
            prog_ph.progress(
                min(max(pct, 0.0), 1.0),
                text="Frame {} / {}  |  {}  |  UniqueThisVideo: {}".format(
                    idx, total,
                    "DET" if fstats["detection_frame"] else "skip",
                    st.session_state["video_unique"],
                ),
            )

        # Count entries/exits from DB for this session (approximation using log)
        lines = logger_utils.read_events_log(st.session_state["session_logs_dir"])
        entries = sum(1 for l in lines if "ENTRY" in l)
        exits   = sum(1 for l in lines if "EXIT"  in l)
        st.session_state["video_entries"] = entries
        st.session_state["video_exits"]   = exits

        stat_ph.markdown(
            "**Active:** {}  |  **Unique this video:** {}  |  "
            "**Entries:** {}  |  **Exits:** {}".format(
                st.session_state["active_count"],
                st.session_state["video_unique"],
                entries, exits,
            )
        )
        # Rerun immediately to process next batch
        st.rerun()

    # ── Done state ────────────────────────────────────────────────────
    if st.session_state["done"]:
        idm = st.session_state.get("idm")
        vid_unique = idm.get_video_unique_count() if idm else st.session_state["video_unique"]
        st.success(
            "✅ Done!  Unique visitors this video: **{}**  |  "
            "Entries: {}  |  Exits: {}".format(
                vid_unique,
                st.session_state["video_entries"],
                st.session_state["video_exits"],
            )
        )
        if st.button("Process another video"):
            st.session_state["done"] = False
            st.session_state["processing"] = False
            st.rerun()


# ─────────────────────────────────────────────────────────────────────
def render_db_tab(cfg: dict):
    """Issue 4: shows all-time DB data for reference."""
    st.markdown("### Face Table  *(all videos)*")
    faces = db_manager.get_all_faces(cfg["db_path"])
    if faces:
        st.dataframe(pd.DataFrame([{
            "face_id":  r["face_id"][:12]+"...",
            "entry":    r["entry_timestamp"],
            "exit":     r["exit_timestamp"] or "-",
            "face_emb": "Y" if r["face_embedding"] is not None else "N",
            "body_emb": "Y" if r["body_embedding"] is not None else "N",
        } for r in faces]), use_container_width=True)
    else:
        st.info("No records yet.")

    st.markdown("### Body Table  *(all videos)*")
    bodies = db_manager.get_all_bodies(cfg["db_path"])
    if bodies:
        st.dataframe(pd.DataFrame([{
            "body_id":     r["body_id"][:12]+"...",
            "entry":       r["entry_timestamp"],
            "exit":        r["exit_timestamp"] or "-",
            "linked_face": r["linked_face_id"][:8]+"..." if r["linked_face_id"] else "-",
        } for r in bodies]), use_container_width=True)
    else:
        st.info("No records yet.")

    st.markdown("**All-time unique count (DB): `{}`**".format(
        db_manager.get_unique_count(cfg["db_path"])
    ))
    if st.button("Refresh DB"):
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
def render_log_tab(cfg: dict):
    """
    Issue 1b fix: show THIS video's events.log, not the global one.
    Falls back to listing all available session logs if no active session.
    """
    session_dir = st.session_state.get("session_logs_dir")

    if session_dir and os.path.isdir(session_dir):
        st.markdown("**Log for:** `{}`".format(os.path.basename(session_dir)))
        lines = logger_utils.read_events_log(session_dir, last_n=200)
        if lines:
            parts = []
            for line in reversed(lines):
                color = "#00e676" if "ENTRY" in line else "#ef5350"
                case_badge = ""
                if "case=face"    in line: case_badge = " 🟢"
                elif "case=noclear" in line: case_badge = " 🟡"
                elif "case=body"  in line: case_badge = " 🔵"
                parts.append(
                    "<span style=\"color:{c}\">{t}{b}</span>".format(
                        c=color, t=line, b=case_badge)
                )
            st.markdown(
                '<div class="log-box">{}</div>'.format("<br>".join(parts)),
                unsafe_allow_html=True,
            )
        else:
            st.info("No events yet for this video.")
    else:
        # Show list of all past session logs
        base = cfg["logs_dir"]
        if os.path.isdir(base):
            sessions = sorted(
                [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
                reverse=True,
            )
            if sessions:
                selected = st.selectbox("Select a past session log", sessions)
                sel_dir  = os.path.join(base, selected)
                lines    = logger_utils.read_events_log(sel_dir, last_n=200)
                if lines:
                    parts = []
                    for line in reversed(lines):
                        color = "#00e676" if "ENTRY" in line else "#ef5350"
                        parts.append(
                            "<span style=\"color:{c}\">{t}</span>".format(c=color, t=line)
                        )
                    st.markdown(
                        '<div class="log-box">{}</div>'.format("<br>".join(parts)),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Empty log.")
            else:
                st.info("No session logs yet.")
        else:
            st.info("No logs directory found.")

    if st.button("Refresh Log"):
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
def main():
    cfg = load_config()
    init_system(cfg)
    init_session_state()
    render_sidebar(cfg)

    st.markdown("# 👁 Intelligent Face Tracker")
    st.markdown(
        "**YOLOv8** · **ByteTrack** · "
        "**InsightFace ArcFace** · **ResNet50** body · **SQLite**"
    )
    st.markdown("---")

    tab_live, tab_db, tab_log = st.tabs(
        ["📹 Live Tracking", "🗄 Database", "📋 Events Log"]
    )
    with tab_live: render_live_tab(cfg)
    with tab_db:   render_db_tab(cfg)
    with tab_log:  render_log_tab(cfg)


if __name__ == "__main__":
    main()