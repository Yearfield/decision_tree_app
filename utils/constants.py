# utils/constants.py

# Columns for the decision tree levels
NODE_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]

# Alias kept for backward compatibility wherever LEVEL_COLS was used
LEVEL_COLS = NODE_COLS

# Human-friendly labels (VM + nodes)
LEVEL_LABELS = ["Vital Measurement", *NODE_COLS]

# Canonical column order expected in sheets
CANON_HEADERS = ["Vital Measurement", *NODE_COLS, "Diagnostic Triage", "Actions"]

# Maximum tree depth and children per parent
MAX_LEVELS = 5
MAX_CHILDREN_PER_PARENT = 5

# Root label used by logic/materialization (keep existing value if already defined elsewhere)
ROOT_PARENT_LABEL = "<ROOT>"

# Additional constants for backward compatibility
APP_VERSION = "v8.0.0"
ROOT_COL = "Vital Measurement"  # VM / root
ROOT_LEVEL = 0
NODE_LEVELS = 5

# UI strings
TAB_ICONS = {
    "source":"📂","workspace":"🗂","validation":"🔎","conflicts":"⚖️","triage":"🩺","actions":"⚡",
    "symptoms":"🧬","dictionary":"📖","calculator":"🧮","visualizer":"🌐","push_log":"📜"
}
