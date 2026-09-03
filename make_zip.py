"""
Builds clean zip archive of langgraph-interview-agent excluding cache files.
"""

import os
import zipfile
from pathlib import Path

source_dir = Path(r"C:\Users\ujjwa\.gemini\antigravity\scratch\langgraph-interview-agent")
target_zip = Path(r"C:\Users\ujjwa\.gemini\antigravity\scratch\langgraph-interview-agent.zip")

exclude_patterns = ["__pycache__", ".pytest_cache", ".git", ".venv", ".ds_store", "qdrant"]

print(f"Creating zip archive from {source_dir} to {target_zip}...")
with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(source_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if not any(p in d.lower() for p in exclude_patterns)]
        for file in files:
            if any(p in file.lower() for p in exclude_patterns):
                continue
            file_path = Path(root) / file
            arcname = file_path.relative_to(source_dir)
            zf.write(file_path, arcname)

size_mb = target_zip.stat().st_size / (1024 * 1024)
print(f"Successfully created zip: {target_zip} ({size_mb:.2f} MB)")
