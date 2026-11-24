"""House style for generated tables that have no group rows or spanning headers of their own.

The main tables and most supplementary tables carry shaded group headings, header bands or tinted
CyteArc rows (colours `tableband`, `tablezebra` and `cytearc`, defined in preamble.tex with
colortbl). A table without such structure gets the same look at its top: every header row (between
\\toprule and the first \\midrule) on the `tableband` band, and, for long uniform tables, every other
body row on `tablezebra`. Column gaps are set to 3 pt, as in the main tables, because colortbl paints
shaded rows `\\tabcolsep` beyond the outer edges of a table whose column specification starts and ends
with @{}. Only formatting changes; cells, rules and row order are kept.
"""

import re

ROW_END = re.compile(r"\\\\\s*(\[[^\]]*\])?\s*$")


def house_style(text: str, *, zebra: bool = False) -> str:
    out, state, body_rows = [], "pre", 0
    for line in text.split("\n"):
        s = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if state == "pre" and s.startswith("\\toprule"):
            state = "head"
        elif state == "head" and s.startswith("\\midrule"):
            state = "body"
        elif state == "body" and s.startswith("\\bottomrule"):
            state = "post"
        elif (
            ROW_END.search(s)
            and "\\rowcolor" not in s
            and not s.startswith(("\\cmidrule", "\\addlinespace"))
        ):
            if state == "head":
                line = f"{indent}\\rowcolor{{tableband}}{s}"
            elif state == "body":
                if zebra and body_rows % 2 == 1:
                    line = f"{indent}\\rowcolor{{tablezebra}}{s}"
                body_rows += 1
        out.append(line)
    styled = "\n".join(out)
    if "\\setlength{\\tabcolsep}" not in styled:
        styled = "\\sffamily\\setstretch{1}\\color{ink}\\setlength{\\tabcolsep}{3pt}\n" + styled
    return styled
