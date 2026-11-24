import csv
import subprocess
from itertools import groupby

from manuscript.tables.render import OUTPUT, TABLES

STEM = "table1_approaches"


def escape(text: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in text)


def mark(token: str) -> str:
    kind, note = token[0], token[1:]
    if kind not in ("Y", "P", "N", "-"):
        raise ValueError(f"unknown capability mark: {token}")
    if kind == "-":
        symbol = r"{\color{muted}n/a}"
    else:
        shape = r"\draw[ink,line width=.35pt] (0,0) circle (1.05mm);"
        if kind == "Y":
            shape = r"\fill[ink] (0,0) circle (1.05mm);"
        elif kind == "P":
            shape += r"\fill[ink] (0,0)--(90:1.05mm) arc (90:270:1.05mm)--cycle;"
        symbol = r"\tikz[baseline=-.55ex]{" + shape + "}"
    return (
        r"\strut "
        + symbol
        + (rf"\rlap{{\textsuperscript{{\color{{slate}}\itshape {escape(note)}}}}}" if note else "")
    )


def latex() -> str:
    with (TABLES / "table_1" / "data.csv").open(newline="") as handle:
        data = list(csv.DictReader(handle))
    if not data or data[0]["capability"] != "Tools":
        raise ValueError("the first capability row must list the tools")
    approaches = [key for key in data[0] if key not in ("group", "capability")]
    if len(approaches) != 6 or approaches[-1] != "CyteArc":
        raise ValueError("Table 1 requires six approaches ending with CyteArc")
    header = [r"\textbf{Capability}"] + [r"\textbf{" + escape(name) + "}" for name in approaches]
    tools = [r"{\itshape Tools of each approach}"] + [
        r"{\fontsize{6}{7.2}\selectfont\color{slate}"
        + r"\parbox[t]{18.5mm}{\centering "
        + r"\par ".join(escape(tool.strip()) for tool in data[0][name].split(","))
        + "}}"
        for name in approaches
    ]
    body = [" & ".join(header) + r" \\[1.4mm]\hline", " & ".join(tools) + r" \\[1.2mm]\hline"]
    for group, items in groupby(data[1:], lambda row: row["group"]):
        body.append(
            r"\rowcolor{tableband}\multicolumn{6}{l}{\strut\textbf{" + escape(group) + r"}} & \\"
        )
        for row in items:
            body.append(
                escape(row["capability"])
                + " & "
                + " & ".join(mark(row[name]) for name in approaches)
                + r" \\[.75mm]\arrayrulecolor{rule}\hline\arrayrulecolor{ink}"
            )
    legend = r"\hspace{4mm}".join(
        mark(token) + " " + text
        for token, text in (
            ("Y", "supported in that workflow"),
            ("P", "partial, or differs between the tools of a column"),
            ("N", "not provided"),
            ("-", "outside the role of the approach"),
        )
    )
    return (
        r"""\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros,array,colortbl,tikz}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{ink}{HTML}{141414}
\definecolor{slate}{HTML}{555E68}
\definecolor{muted}{HTML}{9AA3AD}
\definecolor{tableband}{HTML}{F0F0F0}
\definecolor{cytearc}{HTML}{E5F0F7}
\definecolor{rule}{HTML}{E5E8EC}
\setlength{\tabcolsep}{0pt}
\begin{document}
\begin{minipage}{160mm}
\color{ink}\fontsize{7}{8.3}\selectfont
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{49mm}*{5}{>{\centering\arraybackslash}p{18.5mm}}>{\columncolor{cytearc}\centering\arraybackslash}p{18.5mm}@{}}
"""
        + "\n".join(body)
        + r"""
\end{tabular}
\par\vspace{1.8mm}
{\fontsize{6.2}{7.4}\selectfont\color{slate}
"""
        + legend
        + r"""}
\end{minipage}
\end{document}
"""
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = OUTPUT / f"{STEM}.tex"
    source.write_text(latex())
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
        cwd=OUTPUT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


if __name__ == "__main__":
    main()
