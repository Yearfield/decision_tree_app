# Decision Tree Builder (Streamlit)

A modular Streamlit app to build, validate, and visualize hierarchical decision trees sourced from Excel/CSV or Google Sheets. It helps authors normalize branches (exactly 5 options per parent), detect conflicts, validate logic, and export artifacts for downstream use.

## Features

- Upload XLSX/CSV or connect a Google Sheet
- Workspace selection to focus on a sheet
- Symptoms editor with auto-cascade and 5-option enforcement
- Conflict detection and resolution presets
- Dictionary view for global label operations
- Validation report (orphans, loops, red-flag coverage)
- Calculator for option distributions and “red flag” coverage
- Visualizer (PyVis) for interactive tree graphs
- Push Log for auditing export/push operations

## Quickstart

1) Install
- Python 3.10+ (3.12 recommended)
- Create venv and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Configure Google Sheets
- Create `.streamlit/secrets.toml` (see “Google Sheets config”)

3) Run the app
```bash
streamlit run streamlit_app_upload.py
```
- Open http://localhost:8501

## Data schema

The app expects the first 8 columns in this exact order:

- Parent/child logic is computed from `Node 1..5`
- “Vital Measurement” partitions independent trees (e.g., BP, HR, etc.)

## Repository layout

- `streamlit_app_upload.py` — main app: tabs and navigation
- `utils.py` — single source of truth for constants and helpers:
  - `CANON_HEADERS`, `LEVEL_COLS`, `MAX_LEVELS`
  - normalization, header validation, path/key helpers, enforce_k_five
- UI modules
  - `ui_source.py` — upload (XLSX/CSV), Google Sheets load, workbook builder
  - `ui_workspace.py` — pick active sheet, normalization helpers
  - `ui_symptoms.py` — browse/edit branch children with auto-cascade
  - `ui_conflicts.py` — detect/resolve branching conflicts
  - `ui_dictionary.py` — label dictionary and bulk operations
  - `ui_validation.py` — validation flows and UX wiring
  - `ui_calculator.py` — distributions and red-flag coverage
  - `ui_visualizer.py` — PyVis graph
  - `ui_pushlog.py` — session push log
- Logic modules
  - `logic_cascade.py` — cascade/anchor-reuse engines
  - `logic_conflicts.py` — conflict detection utilities
  - `logic_export.py` — Google Sheets read/write/backup, CSV/XLSX/JSON export
  - `logic_validation.py` — validation algorithms (new/combined/legacy APIs)
- Docs
  - `EXCEPTION_HANDLING_REFACTOR.md` — exception-handling decisions and coverage
  - `CONSTANTS_CONSOLIDATION_SUMMARY.md` — constants centralization details
- Backups: `streamlit_app_upload_v*_backup.py` (historical references)

## Google Sheets config (optional)

Place a service account JSON under `[gcp_service_account]` in `.streamlit/secrets.toml`:
```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "svc-account@your-project.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```
- The app will show “Google Sheets linked ✓” in the header when detected.
- Sheets access uses `gspread` with Drive/Spreadsheets scopes.

## Usage

1) Source
- Upload an `.xlsx`/`.csv` or load a Google Sheet
- Optionally create an in-session sheet (blank scaffolding)

2) Workspace
- Pick the workbook and the active sheet to work on
- Normalize labels/casing if desired

3) Symptoms
- Browse parents (by level) and edit child sets
- Enforce 5 options via editor or presets
- Auto-cascade: fill anchor rows or append new rows as needed

4) Conflicts
- Detect where identical parent labels define different child sets
- Resolve by keeping a selected set or custom set (trim/pad to 5)

5) Dictionary
- Inspect and bulk-edit labels; synonyms, casing, merges

6) Validation
- Run checks (orphans, loops, missing red-flag coverage)
- Download JSON report

7) Calculator
- Explore distributions per parent path and red-flag coverage

8) Visualizer
- Inspect the tree graph; filter by Vital Measurement; toggle merged labels

9) Push Log
- Review session push operations and export to CSV

## Export & integrations

- CSV/XLSX bytes export: `logic_export.export_dataframe_to_csv_bytes`, `export_dataframe_to_excel_bytes`
- Overrides JSON export/import for branch sets
- Google Sheets:
  - Read tab to DataFrame
  - Overwrite tab (with resize), or create if missing
  - Create timestamped backup tabs

## Development notes

- Constants live in `utils.py` (import instead of redefining)
- Type hints used across modules
- Exception handling: most `except Exception` replaced with specific exceptions (see `EXCEPTION_HANDLING_REFACTOR.md`)
- Streamlit-only code is kept in UI modules; logic modules avoid Streamlit deps
- Graph: `pyvis` used for visualizations

## Dependencies

Pinned by capability in `requirements.txt`, including:
- `streamlit`, `pandas`, `numpy`, `openpyxl`, `gspread`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`
- Optional UI/data helpers: `markdown-it-py`, `python-dateutil`, `fuzzywuzzy[speedup]`, `pyvis`, `networkx`

Install with:
```bash
pip install -r requirements.txt
```

## Testing

- Add `pytest`-based tests under `tests/`
- Suggested targets:
  - `utils.py` (normalization, header checks, enforce_k_five)
  - `logic_cascade.py` (store building, anchor reuse paths)
  - `logic_conflicts.py` (conflict detection & resolution)
  - `logic_export.py` (JSON import/export; mock Google APIs)

## Troubleshooting

- “Headers mismatch”: ensure the first 8 columns match the canonical schema
- “No secrets files found”: create `.streamlit/secrets.toml` as shown above
- Visualizer shows no nodes: check filters and data completeness
- Google Sheets failures: verify service account access to the spreadsheet and tab names
- Running as plain Python raises Streamlit warnings — always launch via `streamlit run`

## Security

- `.streamlit/secrets.toml` is ignored by Git (`.gitignore`)
- Never commit private keys or service account files

## License

Proprietary or to be determined. Update this section if you intend to open-source.

## Acknowledgements

Built with:
- Streamlit, Pandas, NumPy
- gspread & Google Auth libraries
- PyVis / NetworkX (visualization)