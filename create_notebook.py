import nbformat
from nbformat.v4 import new_notebook, new_code_cell

# Read the Python code from the temporary file
with open("temp_explore_redd.py", "r") as f:
    python_code = f.read()

# Create a new Jupyter notebook
nb = new_notebook()

# Add a code cell with the content of the Python script
nb.cells.append(new_code_cell(python_code))

# Write the notebook to a .ipynb file
with open("explore_redd_data.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Jupyter notebook 'explore_redd_data.ipynb' created successfully.")