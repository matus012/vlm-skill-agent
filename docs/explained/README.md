# The explainer PDF

`docs/vlm-skill-agent-explained.pdf` is a long, plain-English explanation of the whole project for a reader who has
never seen it. It is built from the Markdown files in this folder, so it can be rebuilt.

Rebuild it, from the repository root, with the docs environment (fpdf2 and matplotlib, pinned in
`docs/explained/requirements.txt`):

```bash
python docs/explained/build_pdf.py
```

One-time setup of that environment:

```bash
uv venv .venv-docs && uv pip install --python .venv-docs/bin/python -r docs/explained/requirements.txt
.venv-docs/bin/python docs/explained/build_pdf.py
```

Nothing else is needed: the fonts are the DejaVu files bundled with matplotlib, the figures are drawn by `figures.py`
into a temporary folder, so this folder needs nothing from the rest of the repository.

| File | What it is |
|---|---|
| `01-*.md` to `12-*.md` | the numbered sections |
| `A-*.md` to `E-*.md` | the appendices: the full prompt, the skill API, the deviation log, compute used, the report index |
| `figures.py` | the figures (matplotlib) |
| `build_pdf.py` | the renderer: a small Markdown subset to PDF with fpdf2, with table of contents and numbered figures |

Every number in the text traces to a task report of the project. In the development repository a script checks that; it is not part of the public copy.
