import re
import streamlit as st
from typing import Optional, List

# Import helper functions from the app module
from app.sheets import (
	get_gspread_client_from_secrets,
	open_spreadsheet,
	list_worksheets,
	create_worksheet,
	read_worksheet,
	write_dataframe,
)


def extract_spreadsheet_id(value: str) -> str:
	"""Extract spreadsheet ID from either a raw ID or a full Google Sheets URL."""
	if not value:
		raise ValueError("Please provide a Spreadsheet ID or full URL.")

	value = value.strip()

	# If it's already a plausible ID (typically 44 chars, but be lenient)
	if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", value):
		return value

	# Try to extract from common URL formats
	match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]{20,})", value)
	if match:
		return match.group(1)

	raise ValueError("Could not extract Spreadsheet ID. Paste the ID or a valid Google Sheets URL.")


def init_state() -> None:
	"""Initialize keys in session state used by this page."""
	for key, default in [
		("gclient", None),
		("spreadsheet", None),
		("worksheet_names", []),
		("spreadsheet_id_input", ""),
		("selected_worksheet", None),
		("new_worksheet_name", ""),
		("current_df", None),
		("df_loaded", False),
		("save_mode", "overwrite"),
	]:
		if key not in st.session_state:
			st.session_state[key] = default


def connect_and_list() -> None:
	"""Connect using secrets, open the spreadsheet, and list worksheet names."""
	try:
		# Get credentials from st.secrets; never print them
		service_account_info = st.secrets["gcp_service_account"]

		with st.spinner("🔗 Connecting to Google Sheets..."):
			client = get_gspread_client_from_secrets(service_account_info)
			spreadsheet_id = extract_spreadsheet_id(st.session_state["spreadsheet_id_input"])
			ss = open_spreadsheet(client, spreadsheet_id)
			names = list_worksheets(ss)

		# Update session state on success
		st.session_state["gclient"] = client
		st.session_state["spreadsheet"] = ss
		st.session_state["worksheet_names"] = names
		if names:
			st.session_state["selected_worksheet"] = names[0]
		else:
			st.session_state["selected_worksheet"] = None

		# Show success message with sheet info
		st.success(f"✅ Connected to spreadsheet: {spreadsheet_id}")
		if names:
			st.info(f"📋 Found {len(names)} worksheet(s): {', '.join(names)}")
		else:
			st.info("📋 No worksheets found. You can create a new one below.")
			
	except KeyError:
		st.error("❌ Missing gcp_service_account in secrets. Add your service account JSON to st.secrets['gcp_service_account'].")
	except Exception as e:
		# Show a user-friendly error, but do not reveal secrets
		st.error(str(e))


def create_new_worksheet() -> None:
	"""Create a new worksheet with the provided name and refresh list."""
	name = (st.session_state.get("new_worksheet_name") or "").strip()
	if not name:
		st.warning("⚠️ Please enter a name for the new worksheet.")
		return

	if st.session_state.get("spreadsheet") is None:
		st.warning("⚠️ Connect to a spreadsheet first.")
		return

	try:
		with st.spinner(f"📝 Creating worksheet '{name}'..."):
			ws = create_worksheet(st.session_state["spreadsheet"], name)
			# Refresh worksheet list
			names = list_worksheets(st.session_state["spreadsheet"])
			st.session_state["worksheet_names"] = names
			st.session_state["selected_worksheet"] = name
		
		st.success(f"✅ Worksheet '{name}' created successfully!")
	except Exception as e:
		st.error(str(e))


def load_worksheet_data() -> None:
	"""Load the selected worksheet data into a DataFrame."""
	selected_worksheet = st.session_state.get("selected_worksheet")
	if not selected_worksheet:
		st.warning("⚠️ Please select a worksheet first.")
		return

	if st.session_state.get("spreadsheet") is None:
		st.warning("⚠️ Connect to a spreadsheet first.")
		return

	try:
		with st.spinner(f"📥 Loading worksheet '{selected_worksheet}'..."):
			# Load worksheet data using the helper function
			df = read_worksheet(st.session_state["spreadsheet"], selected_worksheet)
			
			# Store in session state
			st.session_state["current_df"] = df
			st.session_state["df_loaded"] = True
		
		if df.empty:
			st.info("📋 Worksheet is empty. You can add headers and data below.")
		else:
			st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns from '{selected_worksheet}'.")
			
	except Exception as e:
		st.error(str(e))


def create_empty_dataframe_with_headers() -> None:
	"""Create an empty DataFrame with column headers for editing."""
	headers = st.text_input(
		"Enter column headers (comma-separated)",
		placeholder="e.g., Name, Age, Email, Phone",
		help="Enter column names separated by commas"
	)
	
	if headers:
		header_list = [h.strip() for h in headers.split(",") if h.strip()]
		if header_list:
			# Create empty DataFrame with headers
			import pandas as pd
			df = pd.DataFrame(columns=header_list)
			st.session_state["current_df"] = df
			st.session_state["df_loaded"] = True
			st.success(f"✅ Created empty DataFrame with {len(header_list)} columns.")
			st.rerun()
		else:
			st.warning("⚠️ Please enter at least one column header.")


def save_dataframe_to_sheet() -> None:
	"""Save the current DataFrame to the selected worksheet."""
	selected_worksheet = st.session_state.get("selected_worksheet")
	current_df = st.session_state.get("current_df")
	save_mode = st.session_state.get("save_mode", "overwrite")
	
	if not selected_worksheet:
		st.error("❌ Please select a worksheet first.")
		return
		
	if current_df is None:
		st.error("❌ No data to save. Load a worksheet first.")
		return
		
	# Validate DataFrame is not empty before saving
	if current_df.empty:
		st.warning("⚠️ Cannot save empty DataFrame. Please add some data first.")
		return
		
	if st.session_state.get("spreadsheet") is None:
		st.error("❌ Not connected to a spreadsheet.")
		return
	
	try:
		with st.spinner(f"💾 Saving data ({save_mode} mode)..."):
			write_dataframe(
				st.session_state["spreadsheet"],
				selected_worksheet,
				current_df,
				mode=save_mode
			)
		
		# Use status for success message
		st.success(f"✅ Data saved successfully to '{selected_worksheet}' ({save_mode} mode).")
		
	except Exception as e:
		st.error(str(e))


def get_csv_download_data() -> str:
	"""Convert current DataFrame to CSV string for download."""
	import pandas as pd
	import io
	
	current_df = st.session_state.get("current_df")
	if current_df is None or current_df.empty:
		return ""
	
	# Convert DataFrame to CSV string
	output = io.StringIO()
	current_df.to_csv(output, index=False)
	return output.getvalue()


# ---- Page UI ----
st.set_page_config(page_title="Google Sheet Editor", page_icon="📄", layout="wide")
init_state()

st.title("Google Sheet Editor")

# Spreadsheet ID / URL input
st.session_state["spreadsheet_id_input"] = st.text_input(
	"Spreadsheet ID or URL",
	value=st.session_state.get("spreadsheet_id_input", ""),
	placeholder="Paste a Google Sheets URL or the Spreadsheet ID",
)

# Connect button
if st.button("Connect & List Tabs", type="primary"):
	connect_and_list()

# After connection, show worksheet controls
if st.session_state.get("spreadsheet") is not None:
	with st.container(border=True):
		st.subheader("Worksheets")
		names = st.session_state.get("worksheet_names", [])
		if names:
			st.session_state["selected_worksheet"] = st.selectbox(
				"Select a worksheet",
				names,
				index=names.index(st.session_state.get("selected_worksheet")) if st.session_state.get("selected_worksheet") in names else 0,
			)
		else:
			st.info("No worksheets found. Create one below.")

		# New worksheet creation
		st.session_state["new_worksheet_name"] = st.text_input(
			"New worksheet name",
			value=st.session_state.get("new_worksheet_name", ""),
			placeholder="Enter a unique worksheet name",
		)
		if st.button("Create worksheet"):
			create_new_worksheet()

		# Load worksheet data
		if st.button("Load worksheet", type="secondary"):
			load_worksheet_data()

# Data editing section
if st.session_state.get("df_loaded") and st.session_state.get("current_df") is not None:
	current_df = st.session_state["current_df"]
	
	st.subheader("Edit Data")
	
	# Always show data summary prominently
	with st.container(border=True):
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Rows", len(current_df))
		with col2:
			st.metric("Columns", len(current_df.columns))
		with col3:
			st.metric("Status", "Empty" if current_df.empty else "Has Data")
	
	# Display the DataFrame in an editable grid
	edited_df = st.data_editor(
		current_df,
		use_container_width=True,
		editable=True,
		num_rows="dynamic",
	)
	
	# Update session state with edited DataFrame
	st.session_state["current_df"] = edited_df
	
	# Save controls section
	st.subheader("Save Data")
	
	# Save mode selection
	st.session_state["save_mode"] = st.radio(
		"Save mode:",
		options=["overwrite", "append"],
		format_func=lambda x: "Overwrite worksheet" if x == "overwrite" else "Append rows",
		horizontal=True,
		index=0 if st.session_state.get("save_mode", "overwrite") == "overwrite" else 1
	)
	
	# Download and Save buttons in columns
	col1, col2 = st.columns(2)
	
	with col1:
		# CSV download button
		csv_data = get_csv_download_data()
		if csv_data:
			st.download_button(
				label="📥 Download CSV backup",
				data=csv_data,
				file_name=f"{st.session_state.get('selected_worksheet', 'data')}.csv",
				mime="text/csv",
				help="Download current data as CSV before saving"
			)
		else:
			st.button("📥 Download CSV backup", disabled=True, help="No data to download")
	
	with col2:
		# Save button with validation
		if edited_df.empty:
			st.button("💾 Save changes", disabled=True, help="Cannot save empty DataFrame")
		else:
			if st.button("💾 Save changes", type="primary"):
				save_dataframe_to_sheet()

elif st.session_state.get("df_loaded") and st.session_state.get("current_df") is not None and st.session_state["current_df"].empty:
	st.subheader("Empty Worksheet")
	st.info("📋 This worksheet is empty. You can create headers and start adding data.")
	
	# Option to create headers for empty worksheet
	create_empty_dataframe_with_headers()
