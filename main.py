import os
import jupytext
import nbformat

md_sections = [
    "intro/intro.md",
    "intro/problem_statement.md",
    "intro/existing_solutions.md",
    "model/model.md",
]

def create_notebook():
    notebook = None

    for md_section in md_sections:
        section = jupytext.read(md_section)   # read the file once

        if notebook is None:
            notebook = section
        else:
            notebook.cells.extend(section.cells)

    nbformat.write(notebook, "project.ipynb")

if __name__ == "__main__":
    create_notebook()