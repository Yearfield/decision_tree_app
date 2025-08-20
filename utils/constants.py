# utils/constants.py
APP_VERSION = "v6.0.0"
CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions"
]
LEVEL_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
MAX_LEVELS = 5
MAX_CHILDREN_PER_PARENT = 5
ROOT_PARENT_LABEL = "<ROOT>"  # synthetic parent for Level 1
TAB_ICONS = {
    "source":"📂","workspace":"🗂","validation":"🔎","conflicts":"⚖️","triage":"🩺","actions":"⚡",
    "symptoms":"🧬","dictionary":"📖","calculator":"🧮","visualizer":"🌐","push_log":"📜"
}
