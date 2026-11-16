"""Build docs/vlm-skill-agent-explained.pdf from the Markdown files next to this script.

    python docs/explained/build_pdf.py            (any venv with the pins in docs/explained/requirements.txt)

Sections are the files 01-*.md ... 12-*.md, appendices A-*.md ... E-*.md. The Markdown is a small subset:
# section title, ## subsection, paragraphs, - bullets, 1. lists, | tables |, > callout, ``` code ```,
![caption](fig:name) for a figure drawn by figures.py. Fonts come from matplotlib's bundled DejaVu files and the figures
are drawn into a temporary folder, so a fresh clone needs nothing else.
"""
from __future__ import annotations

import re
import tempfile
import textwrap
from pathlib import Path

import matplotlib
from fpdf import FPDF
from fpdf.fonts import FontFace
from PIL import Image

import figures

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "vlm-skill-agent-explained.pdf"
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

TITLE = "A vision-language model that drives robots from photos"
SUBTITLE = "The vlm-skill-agent project explained: system, evaluation and lessons"
AUTHOR = "Matus Filo"
DATE = "2026-11-08"
ABSTRACT = ("One vision-language model, one prompt and one small agent drive two very different simulated robots, a wheeled "
            "TurtleBot 4 and a walking Unitree G1 humanoid, from camera photos alone. This document explains the whole "
            "project for a reader who has never seen it: the system, the simulated clinic, the skills, how the tests were "
            "scored and kept honest, the results on both bodies, everything that went wrong, and the limits. "
            "Simulation only.")

INK, GREY, BLUE, LIGHT = (31, 41, 51), (107, 119, 133), (37, 99, 168), (238, 242, 247)
BODY, LH = 8.9, 4.3           # body font size (pt) and line height (mm)
LM = RM = 18.5
FOOTNOTE = "† The number comes from a task report that was not independently audited (section 1.5)."


# ---------------------------------------------------------------- markdown subset
def parse(text: str) -> list[tuple]:
    lines, i, out = text.splitlines(), 0, []
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append(("code", lines[i + 1:j])); i = j + 1
        elif ln.startswith("## "):
            out.append(("h2", ln[3:].strip().replace("`", ""))); i += 1
        elif ln.startswith("# "):
            out.append(("h1", ln[2:].strip().replace("`", ""))); i += 1
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*)\]\(fig:(\w+)\)", ln)
            out.append(("fig", m.group(1), m.group(2))); i += 1
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip().replace("\x00", "|") for c in lines[i].replace("\\|", "\x00").strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(("table", rows))
        elif ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip()); i += 1
            out.append(("quote", " ".join(buf)))
        elif re.match(r"(- |\d+\. )", ln):
            kind, items = ("ul" if ln.startswith("- ") else "ol"), []
            while i < len(lines) and (re.match(r"(- |\d+\. )", lines[i]) or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(r"(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]).strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append((kind, items))
        else:
            buf = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|```|\||>|!\[|- |\d+\. )", lines[i]):
                buf.append(lines[i].strip()); i += 1
            out.append(("p", " ".join(buf)))
    return out


def plain_md(t: str) -> str:
    """Inline markup for fpdf2's own markdown (bold ** and italic __), backticks dropped."""
    return re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"__\1__", t).replace("`", "")


TOKEN = re.compile(r"(\*\*.+?\*\*|`.+?`|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))")


# ---------------------------------------------------------------- pdf
class Doc(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("D", "", 7.5); self.set_text_color(*GREY)
        self.set_xy(LM, 9); self.cell(self.epw, 4, TITLE)
        self.set_draw_color(200, 205, 212); self.line(LM, 14, 210 - RM, 14)
        self.set_text_color(*INK); self.set_y(20)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-13); self.set_font("D", "", 7.5); self.set_text_color(*GREY)
        self.cell(self.epw, 4, f"vlm-skill-agent explained    page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(*INK)

    # -- inline text ---------------------------------------------------
    def inline(self, text: str, size: float = BODY, lh: float = LH, color=INK) -> None:
        self.set_text_color(*color)
        for tok in TOKEN.split(text):
            if not tok:
                continue
            if tok.startswith("**"):
                self.set_font("D", "B", size); tok = tok[2:-2]
            elif tok.startswith("`"):
                self.set_font("M", "", size - 0.9); tok = tok[1:-1]
            elif tok.startswith("*"):
                self.set_font("D", "I", size); tok = tok[1:-1]
            else:
                self.set_font("D", "", size)
            self.write(lh, tok)
        self.set_font("D", "", size)

    def need(self, h: float) -> None:
        if self.get_y() + h > self.h - 18:
            self.add_page()

    def paragraph(self, text: str, indent: float = 0.0, lead: str = "") -> None:
        self.need(LH * 2)
        if lead:
            self.set_font("D", "", BODY); self.set_x(LM + indent - 5.5); self.write(LH, lead)
        self.set_left_margin(LM + indent)
        self.set_x(LM + indent)
        self.inline(text)
        self.ln(LH)
        self.set_left_margin(LM)
        self.set_x(LM)

    def callout(self, text: str) -> None:
        self.set_font("D", "", BODY - 0.2)
        w = self.epw - 6
        h = self.multi_cell(w, LH - 0.1, plain_md(text), markdown=True, align="L", dry_run=True, output="HEIGHT") + 4
        self.need(h + 3)
        y = self.get_y()
        self.set_fill_color(*LIGHT); self.rect(LM, y, self.epw, h, style="F")
        self.set_fill_color(*BLUE); self.rect(LM, y, 1.2, h, style="F")
        self.set_xy(LM + 4, y + 2)
        self.multi_cell(w - 1, LH - 0.1, plain_md(text), markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(255, 255, 255)
        self.set_y(y + h + 3)

    def code(self, lines: list[str]) -> None:
        size = 7.3
        self.set_font("M", "", size)
        cw = self.get_string_width("M")
        width = int((self.epw - 4) / cw)
        out: list[str] = []
        for ln in lines:
            out += textwrap.wrap(ln, width, subsequent_indent="  ", replace_whitespace=False, drop_whitespace=False) or [""]
        lh = 3.5
        self.set_fill_color(*LIGHT)
        for k, ln in enumerate(out):
            self.need(lh + 1)
            self.set_x(LM)
            self.cell(self.epw, lh, " " + ln, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("D", "", BODY)
        self.ln(3)

    def render_table(self, rows: list[list[str]]) -> None:
        n = len(rows[0])
        size = 7.8 if n <= 5 else 7.3
        self.set_font("D", "", size)
        floor, avg = [], []
        for c in range(n):
            col = [r[c] if c < len(r) else "" for r in rows]
            words = max((len(re.sub(r"[`*]", "", w)) for t in col for w in t.split()), default=1)
            floor.append(words * 1.75 + 2.6)
            avg.append(sum(len(t) for t in col) / len(col) + 1)
        total = self.epw
        if sum(floor) > total:
            k = total / sum(floor); floor = [f * k for f in floor]
        extra = total - sum(floor)
        widths = [f + extra * a / sum(avg) for f, a in zip(floor, avg)]
        self.need(20)
        self.set_fill_color(255, 255, 255)
        with super().table(col_widths=tuple(widths), text_align="LEFT", line_height=size * 0.46, markdown=True,
                           borders_layout="HORIZONTAL_LINES", padding=(0.9, 1.2), num_heading_rows=1,
                           headings_style=FontFace(emphasis="BOLD", fill_color=LIGHT), v_align="TOP") as t:
            for r in rows:
                row = t.row()
                for c in r:
                    row.cell(plain_md(c))
        self.set_font("D", "", BODY)
        self.ln(3.5)


def build() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="explained_figs_"))
    figs = figures.make_all(tmp)
    files = sorted(HERE.glob("[0-9][0-9]-*.md")) + sorted(HERE.glob("[A-E]-*.md"))
    pdf = Doc(orientation="P", unit="mm", format="A4")
    pdf.set_margins(LM, 20, RM); pdf.set_auto_page_break(True, 18)
    for style, f in [("", "DejaVuSans.ttf"), ("B", "DejaVuSans-Bold.ttf"), ("I", "DejaVuSans-Oblique.ttf"), ("BI", "DejaVuSans-BoldOblique.ttf")]:
        pdf.add_font("D", style, str(TTF / f))
    pdf.add_font("M", "", str(TTF / "DejaVuSansMono.ttf"))
    pdf.set_title(TITLE); pdf.set_creator("docs/explained/build_pdf.py (fpdf2)")
    pdf.alias_nb_pages()

    # title page
    pdf.add_page()
    pdf.set_y(62); pdf.set_font("D", "B", 27); pdf.set_text_color(*INK)
    pdf.multi_cell(pdf.epw, 11.5, TITLE, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3); pdf.set_font("D", "", 13); pdf.set_text_color(*BLUE)
    pdf.multi_cell(pdf.epw, 6.5, SUBTITLE, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10); pdf.set_font("D", "B", 11); pdf.set_text_color(*INK); pdf.cell(pdf.epw, 6, AUTHOR, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("D", "", 9.5); pdf.set_text_color(*GREY)
    pdf.cell(pdf.epw, 5, f"{DATE}   |   repository: vlm-skill-agent   |   simulation study", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(14); pdf.set_text_color(*INK); pdf.set_font("D", "", 10.5)
    pdf.multi_cell(pdf.epw, 5.6, ABSTRACT, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(250); pdf.set_font("D", "", 8); pdf.set_text_color(*GREY)
    pdf.multi_cell(pdf.epw, 4, "Numbers marked † come from task reports that were not independently audited; "
                   "all other numbers come from audited reports. See section 1.5 and appendix E.", new_x="LMARGIN", new_y="NEXT")

    # table of contents (filled in after the sections are known)
    def render_toc(p: FPDF, outline) -> None:
        p.set_font("D", "B", 17); p.set_text_color(*INK); p.set_xy(LM, 24)
        p.cell(p.epw, 9, "Contents", new_x="LMARGIN", new_y="NEXT")
        entries = list(outline)
        half = (len(entries) + 1) // 2
        while half < len(entries) and entries[half].level != 0:   # start the second column at a section
            half += 1
        colw, y0 = (p.epw - 6) / 2, 38
        for ci, col in enumerate([entries[:half], entries[half:]]):
            x0, y = LM + ci * (colw + 6), y0
            for s_ in col:
                lvl = s_.level
                size = 8.2 if lvl == 0 else 7.4
                ind = 0 if lvl == 0 else 3.5
                if lvl == 0:
                    y += 1.6
                num = str(s_.page_number)
                p.set_font("D", "B" if lvl == 0 else "", size)
                nwid = p.get_string_width(num) + 1
                avail = colw - ind - nwid - 2
                while p.get_string_width(s_.name) + 1 > avail and size > 5.5:
                    size -= 0.2; p.set_font("D", "B" if lvl == 0 else "", size)
                tw = p.get_string_width(s_.name) + 1
                link = p.add_link(page=s_.page_number)
                p.set_xy(x0 + ind, y); p.cell(tw, 4, s_.name, link=link)
                p.set_text_color(*GREY)
                dw = colw - nwid - ind - tw
                p.cell(dw, 4, "." * int(dw / p.get_string_width(".")), link=link); p.set_text_color(*INK)
                p.set_xy(x0 + colw - nwid, y); p.cell(nwid, 4, num, align="R", link=link)
                y += 4.05
    pdf.add_page()
    pdf.insert_toc_placeholder(render_toc, pages=1)

    fig_no, sec_no = 0, 0
    for f in files:
        blocks = parse(f.read_text())
        appendix = not f.name[0].isdigit()
        sec_no += 1
        label = f"Appendix {f.name[0]}" if appendix else str(int(f.name[:2]))
        dagger = any("†" in str(b[1]) for b in blocks)
        for b in blocks:
            k = b[0]
            if k == "h1":
                if f.name[:2] == 'A-':
                    pdf.add_page()
                else:
                    pdf.need(60)
                pdf.ln(2)
                pdf.start_section(f"{label}  {b[1]}" if not appendix else f"{label}. {b[1]}", level=0)
                pdf.set_font("D", "B", 16); pdf.set_text_color(*BLUE)
                pdf.multi_cell(pdf.epw, 7.5, f"{label}  {b[1]}" if not appendix else f"{label}. {b[1]}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(*BLUE); pdf.line(LM, pdf.get_y() + 0.5, 210 - RM, pdf.get_y() + 0.5)
                pdf.ln(4); pdf.set_text_color(*INK); pdf.set_font("D", "", BODY)
            elif k == "h2":
                nxt = blocks[blocks.index(b) + 1] if blocks.index(b) + 1 < len(blocks) else ("p", "")
                if nxt[0] == "fig":
                    pdf.need(Image.open(figs[nxt[2]]).size[1] / Image.open(figs[nxt[2]]).size[0] * 150 + 30)
                else:
                    pdf.need(42 if nxt[0] in ("table", "code") else 28)
                pdf.ln(2.5)
                pdf.start_section(b[1], level=1)
                pdf.set_font("D", "B", 11); pdf.set_text_color(*INK)
                pdf.multi_cell(pdf.epw, 5.5, b[1], new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1); pdf.set_font("D", "", BODY)
            elif k == "p":
                pdf.paragraph(b[1]); pdf.ln(1.6)
            elif k == "quote":
                pdf.callout(b[1])
            elif k == "ul":
                for it in b[1]:
                    pdf.paragraph(it, indent=6, lead="•")
                pdf.ln(1.2)
            elif k == "ol":
                for n, it in enumerate(b[1], 1):
                    pdf.paragraph(it, indent=6, lead=f"{n}.")
                pdf.ln(1.2)
            elif k == "table":
                pdf.render_table(b[1])
            elif k == "code":
                pdf.code(b[1])
            elif k == "fig":
                fig_no += 1
                png = figs[b[2]]
                w_img, h_px = pdf.epw, Image.open(png).size
                w = {"plan": 140, "bodies": 160, "camera": 160}.get(b[2], 165)
                h = w * h_px[1] / h_px[0]
                pdf.need(h + 14)
                pdf.image(str(png), x=LM + (pdf.epw - w) / 2, y=pdf.get_y() + 1, w=w)
                pdf.set_y(pdf.get_y() + h + 2)
                pdf.set_font("D", "I", 8.3); pdf.set_text_color(*GREY)
                pdf.multi_cell(pdf.epw, 4, f"Figure {fig_no}. {b[1]}", align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(*INK); pdf.set_font("D", "", BODY); pdf.ln(3)
        if dagger:
            pdf.set_font("D", "", 7.3); pdf.set_text_color(*GREY)
            pdf.multi_cell(pdf.epw, 3.6, FOOTNOTE, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*INK); pdf.set_font("D", "", BODY)
        pdf.ln(4)
    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    print(build())
