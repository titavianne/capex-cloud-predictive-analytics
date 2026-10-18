import nbformat
from nbclient import NotebookClient
from pathlib import Path
import sys

nb_path = Path("Predictive_Analytics_Hyperscaler_Capex_Cloud_Revenue.ipynb")
print(f"Reading {nb_path}...")
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook...")
try:
    client.execute()
    print("Notebook executed successfully!")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Saved executed notebook with outputs to disk.")
except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)
