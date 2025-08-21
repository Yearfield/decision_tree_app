# utils/constants.py
APP_VERSION = "v6.0.0"
CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions"
]
ROOT_COL = "Vital Measurement"  # VM / root
LEVEL_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]

# Levels: 0..5 (0 = VM, 1..5 = Node 1..5)
ROOT_LEVEL = 0
NODE_LEVELS = 5
MAX_LEVELS = 5  # number of Node columns
MAX_CHILDREN_PER_PARENT = 5
ROOT_PARENT_LABEL = "<ROOT>"

# UI strings
LEVEL_LABELS = ["VM (Root)", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
TAB_ICONS = {
    "source":"📂","workspace":"🗂","validation":"🔎","conflicts":"⚖️","triage":"🩺","actions":"⚡",
    "symptoms":"🧬","dictionary":"📖","calculator":"🧮","visualizer":"🌐","push_log":"📜"
}
