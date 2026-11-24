import csv
import re
from pathlib import Path

from manuscript.tables.style import house_style

TABLES = Path(__file__).resolve().parent
OUTPUT = TABLES.parents[1] / ".build" / "tables"


def read_table(name: str) -> list[dict]:
    with (TABLES / f"{name}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def scalar(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", value):
            return float(value)
        return value

    return [{key: scalar(value) for key, value in row.items()} for row in rows]


def number(value):
    if value == "" or value is None:
        return None
    return float(value)


def read_metrics(name: str) -> dict:
    out = {}
    converters = {
        "int": int,
        "float": float,
        "bool": lambda s: s == "True",
        "str": str,
        "null": lambda s: None,
    }
    with (TABLES / f"{name}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        parts = row["metric"].split(" / ")
        target = out
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        if parts[-1] in target:
            raise ValueError(f"duplicate metric {row['metric']}")
        target[parts[-1]] = converters[row["kind"]](row["value"])
    return out


def write(name: str, text: str, zebra=None) -> None:
    if zebra is not None:
        text = house_style(text, zebra=zebra)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.tex"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    from manuscript.tables import approaches, biology, performance

    performance.render_all()
    biology.render_all()
    approaches.main()


if __name__ == "__main__":
    main()
