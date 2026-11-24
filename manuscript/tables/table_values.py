import csv
import io
import math
from pathlib import Path

TABLES = Path(__file__).resolve().parent


def publication_values(value):
    if isinstance(value, dict):
        result = {str(key): publication_values(item) for key, item in value.items()}
        if len(result) != len(value):
            raise ValueError("publication identifiers collide after string conversion")
        return result
    if isinstance(value, (list, tuple)):
        return [publication_values(item) for item in value]
    if isinstance(value, str):
        return value
    return value.item() if hasattr(value, "item") else value


def write_table(name: str, rows: list[dict]) -> None:
    rows = publication_values(rows)
    if not rows:
        raise ValueError(f"{name}: no table rows")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name}.{key}: non-finite value")
            if isinstance(value, (dict, list, tuple)):
                raise TypeError(f"{name}.{key}: expected a scalar, got {type(value).__name__}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    path = TABLES / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = buffer.getvalue()
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def write_metrics(name: str, values: dict) -> None:
    rows = []

    def visit(value, parts):
        if isinstance(value, dict):
            for key, item in value.items():
                if " / " in str(key):
                    raise ValueError(f"ambiguous metric key: {key}")
                visit(item, [*parts, str(key)])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, [*parts, str(index)])
        else:
            if hasattr(value, "item"):
                value = value.item()
            rows.append(
                {
                    "metric": " / ".join(parts),
                    "value": value,
                    "kind": "null" if value is None else type(value).__name__,
                }
            )

    visit(publication_values(values), [])
    write_table(name, rows)
