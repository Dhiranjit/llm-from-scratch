### Code Analysis Policy

Unless runtime execution is explicitly requested or strictly necessary, always prefer static analysis and mental walkthroughs. Do not spin up virtual environments, run verification scripts, or execute commands just to explain code, trace logic, or answer "what does this do" questions. Inspect and reason through the code statically.

### Jupyter Notebook (.ipynb) Policy

- Avoid raw text/JSON edits on `.ipynb` files whenever possible.
- If modifying a notebook is required, use specialized tooling:
  - Prefer using a dedicated CLI/Python script with `nbformat` (or `jupytext`) to safely read, modify cell source, and serialize back to valid JSON.
  - Alternatively, if exploring or prototyping, convert the notebook to a plain Python script (`jupytext --to py:percent <file>.ipynb`), perform edits cleanly on Python code, and sync it back.
