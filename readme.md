# Decision Tree Builder (Streamlit)

A modular Streamlit app to build, validate, and visualize hierarchical decision trees sourced from Excel/CSV or Google Sheets. It helps authors normalize branches (exactly 5 options per parent), detect conflicts, validate logic, and export artifacts for downstream use.

## Features

- **📂 Source Management**: Upload XLSX/CSV or connect to Google Sheets
- **🗂️ Workspace Selection**: Focus on specific sheets with data quality metrics
- **🧬 Symptoms Editor**: Browse/edit branch children with auto-cascade and 5-option enforcement
- **⚖️ Conflict Resolution**: Detect and resolve branching conflicts with presets
- **📖 Dictionary View**: Global label operations and bulk editing
- **🔎 Validation**: Comprehensive validation (orphans, loops, red-flag coverage)
- **🩺 Diagnostic Triage**: Triage decision management and analysis
- **⚡ Actions Management**: Action definition and bulk operations
- **🌐 Visualizer**: Interactive tree graphs with filtering
- **📜 Push Log**: Audit trail for export/push operations

## Quickstart

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Configure Google Sheets
Create `.streamlit/secrets.toml` with your service account credentials:
```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
token_uri = "https://oauth2.googleapis.com/token"
```

### 3. Run the Application
```bash
streamlit run streamlit_app_upload.py
```
Open http://localhost:8501 in your browser.

## Project Structure & Dependency Boundaries

```
decision_tree_app/
├── streamlit_app_upload.py        # 🚀 Main application entry point
├── monolith.py                    # 📚 Original working monolith (reference)
├── utils/                         # 🎯 Single source of truth
│   ├── __init__.py               # Exports constants and helpers
│   ├── constants.py              # APP_VERSION, CANON_HEADERS, LEVEL_COLS, etc.
│   └── helpers.py                # normalize_text, validate_headers, etc.
├── logic/                         # 🧠 Pure business logic (no Streamlit)
│   ├── __init__.py               # Exports tree and validation functions
│   ├── tree.py                   # infer_branch_options, build_raw_plus_v630, etc.
│   └── validate.py               # detect_orphan_nodes, detect_loops, etc.
├── io_utils/                      # 📡 Pure IO operations (no Streamlit)
│   ├── __init__.py               # Exports Google Sheets and export functions
│   └── sheets.py                 # read_google_sheet, push_to_google_sheets, etc.
├── ui/                           # 🎨 Thin UI renderers
│   ├── __init__.py
│   └── tabs/                     # All UI code in render() functions
│       ├── source.py             # Data upload and VM builder
│       ├── workspace.py          # Sheet selection and preview
│       ├── validation.py         # Validation reports and fixes
│       ├── conflicts.py          # Conflict detection and resolution
│       ├── symptoms.py           # Branch editing and auto-cascade
│       ├── dictionary.py         # Label management
│       ├── triage.py             # Diagnostic triage management
│       ├── actions.py            # Actions management
│       ├── visualizer.py         # Tree visualization
│       └── push_log.py           # Push history and audit
├── tests/                        # 🧪 Pytest test suite
│   ├── test_helpers.py           # Test utils.helpers functions
│   ├── test_tree.py              # Test logic.tree functions
│   ├── test_validate.py          # Test logic.validate functions
│   └── fixtures/
│       └── sample.csv            # Test data
├── scripts/                      # ⚡ Performance and utility scripts
│   ├── gen_synth.py              # Generate 10k synthetic test rows
│   ├── perf_harness.py           # Performance benchmarking
│   └── run_benchmark.py          # Complete benchmark suite
└── synthetic_10k.csv             # Generated performance test data
```

### Dependency Boundaries

- **`utils/`** → No dependencies (pure constants and helpers)
- **`logic/`** → Only `utils/` and standard libraries (pandas, numpy)
- **`io_utils/`** → Only `utils/` and external APIs (gspread, pandas)
- **`ui/tabs/`** → `utils/`, `logic/`, `io_utils/`, and Streamlit
- **`streamlit_app_upload.py`** → `utils/`, `logic/`, `io_utils/`, and `ui/tabs/`

## Data Schema

The app expects exactly 8 columns in this order:

| Column | Description | Example |
|--------|-------------|---------|
| Vital Measurement | Primary vital sign | "Blood Pressure", "Temperature" |
| Node 1 | First decision level | "High", "Low", "Normal" |
| Node 2 | Second decision level | "Severe", "Mild", "Stable" |
| Node 3 | Third decision level | "Immediate", "Monitor", "Regular" |
| Node 4 | Fourth decision level | "Urgent", "Watch", "Standard" |
| Node 5 | Fifth decision level | "Call 911", "Check again", "Follow up" |
| Diagnostic Triage | Triage decision | "Immediate", "Monitor", "Routine" |
| Actions | Action description | "Red flag", "Continue monitoring", "Standard care" |

## Usage Workflow

1. **📂 Source**: Upload XLSX/CSV or load from Google Sheets
2. **🗂️ Workspace**: Select active sheet and preview data quality
3. **🧬 Symptoms**: Edit branch children with auto-cascade
4. **⚖️ Conflicts**: Detect and resolve branching conflicts
5. **📖 Dictionary**: Manage labels and bulk operations
6. **🔎 Validation**: Run comprehensive validation checks
7. **🩺 Triage**: Manage diagnostic triage decisions
8. **⚡ Actions**: Define and manage action items
9. **🌐 Visualizer**: Explore interactive tree graphs
10. **📜 Push Log**: Review export and push history

## Development Notes

### Caching Strategy
- Heavy computations use `@st.cache_data(ttl=600)` with comprehensive cache keys
- Cache keys include: sheet name, APP_VERSION, data shape, and data hash
- Performance tested on 10k-row synthetic data (~4 seconds total)

### Testing
```bash
# Run all tests
pytest -q

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_helpers.py
```

### Performance Testing
```bash
# Generate synthetic data and run benchmarks
python3 scripts/run_benchmark.py

# Generate 10k synthetic rows only
python3 scripts/gen_synth.py

# Run performance benchmarks only
python3 scripts/perf_harness.py
```

### Code Quality
- **Type hints**: Used throughout all modules
- **Docstrings**: All functions documented
- **Linting**: Use `flake8` for code style
- **Error handling**: Specific exceptions, no silent failures
- **Separation of concerns**: UI, logic, and IO strictly separated

### Adding New Features
1. **Constants**: Add to `utils/constants.py`
2. **Helpers**: Add to `utils/helpers.py`
3. **Business Logic**: Add to `logic/tree.py` or `logic/validate.py`
4. **IO Operations**: Add to `io_utils/sheets.py`
5. **UI Components**: Add new tab to `ui/tabs/` with `render()` function
6. **Tests**: Add corresponding test file in `tests/`

## Dependencies

Core dependencies (see `requirements.txt` for full list):
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and analysis
- **gspread**: Google Sheets integration
- **openpyxl**: Excel file handling
- **pytest**: Testing framework
- **pyvis/networkx**: Tree visualization

## Troubleshooting

### Common Issues
- **"Headers mismatch"**: Ensure first 8 columns match canonical schema
- **"No secrets found"**: Create `.streamlit/secrets.toml` with service account
- **"Visualizer shows no nodes"**: Check filters and data completeness
- **"Google Sheets access denied"**: Verify service account permissions
- **"Import errors"**: Ensure virtual environment is activated

### Performance Issues
- **Slow loading**: Check data size and enable caching
- **Memory issues**: Use workspace filters to focus on specific sheets
- **Google Sheets timeouts**: Check network connectivity and API quotas

## Security

- **`.streamlit/secrets.toml`**: Never commit to version control
- **Service accounts**: Use minimal required permissions
- **Data handling**: All data processed locally, no external storage

## Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Follow** the modular architecture and dependency boundaries
4. **Add tests** for new functionality
5. **Run** performance benchmarks for data-heavy features
6. **Submit** a pull request

## License

Proprietary or to be determined. Update this section if you intend to open-source.

## Acknowledgements

Built with:
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **gspread**: Google Sheets integration
- **PyVis/NetworkX**: Tree visualization
- **Pytest**: Testing framework