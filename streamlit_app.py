"""Root Streamlit entrypoint for Streamlit Community Cloud."""

import runpy


# Execute the real Streamlit script as the main module rather than importing it.
# This ensures Streamlit captures all UI elements produced by app/streamlit_app.py.
runpy.run_path("app/streamlit_app.py", run_name="__main__")
