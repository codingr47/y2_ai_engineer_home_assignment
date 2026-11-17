"""Prompt catalog and strict value validation from the shipped v2 taxonomy."""
import json
import math
from pathlib import Path

from src.contracts.http_models import SearchResponse

TAXONOMY = Path(__file__).resolve().parents[1] / "extraction/artifacts/taxonomy_v2.json"
VEHICLE_KEYS = {"יצרנים": "יצרן", "דגמים": "דגם"}
TYPE_MAP = {"מספר": "number", "מחרוזת": "string", "בוליאני": "boolean",
            "תאריך": "date", "תאריך_יחסי_או_מדויק": "date"}


class Catalog:
    def __init__(self, path=TAXONOMY):
        self.source = Path(path).read_bytes()
        taxonomy = json.loads(self.source)
        self.entries = []
        for category, body in taxonomy["קטגוריות"].items():
            self._walk(category, body, {})
        self.hierarchy = {k for e in self.entries for k in e["conditions"]}
        self.hierarchy.update(e["name"] for e in self.entries if e["selector"])

    def _walk(self, category, body, conditions):
        def container(node):
            return isinstance(node, dict) and "טיפוס" not in node and not (
                node and all(isinstance(v, str) for v in node.values()))

        for name, node in body.items():
            kind, values, selector = None, None, False
            if isinstance(node, dict) and "טיפוס" in node:
                kind = TYPE_MAP[node["טיפוס"]]
            elif isinstance(node, list):
                if len(node) == 2 and all(type(v) in (int, float) for v in node):
                    kind = "number"
                else:
                    kind, values = "enum", node
            elif isinstance(node, dict) and node:
                parent = next((k for k, v in body.items() if isinstance(v, list)
                               and set(node) <= set(v)), None) if all(isinstance(v, list) for v in node.values()) else None
                if parent:
                    for value, options in node.items():
                        self.entries.append(dict(category=category, name=name, type="enum", values=options,
                                                 conditions={**conditions, parent: value}, selector=False))
                    continue
                if all(isinstance(v, str) for v in node.values()):
                    continue  # typo mappings are not search filters
                if all(container(v) for v in node.values()):
                    kind, values, selector = "enum", list(node), True
                    for value, child in node.items():
                        self._walk(category, child, {**conditions, name: value})
                else:
                    self._walk(category, node, conditions)
                    continue
            if kind:
                self.entries.append(dict(category=category, name=name, type=kind, values=values,
                                         conditions=dict(conditions), selector=selector))

    def prompt(self):
        lines = []
        for e in self.entries:
            condition = f" [when {json.dumps(e['conditions'], ensure_ascii=False)}]" if e["conditions"] else ""
            values = ": " + " | ".join(map(str, e["values"])) if e["values"] else ""
            lines.append(f"{e['category']}/{e['name']}{condition} ({e['type']}){values}")
        return "\n".join(lines)

    def validate(self, payload):
        result = SearchResponse.model_validate(payload)
        if result.category is None:
            return result
        params = result.params
        for name, value in params.items():
            matches = [e for e in self.entries if e["category"] == result.category and e["name"] == name
                       and all(params.get(k) == v for k, v in e["conditions"].items())]
            if not matches:
                raise ValueError("unknown or inactive field")
            entry = max(matches, key=lambda e: len(e["conditions"]))
            kind = entry["type"]
            if kind == "number":
                numeric = lambda x: type(x) in (int, float) and math.isfinite(x)
                if isinstance(value, dict):
                    if not value or not set(value) <= {"min", "max"} or not all(numeric(v) for v in value.values()):
                        raise ValueError("invalid numeric bounds")
                    if "min" in value and "max" in value and value["min"] > value["max"]:
                        raise ValueError("reversed numeric bounds")
                elif not numeric(value):
                    raise ValueError("invalid number")
            elif kind == "boolean":
                if type(value) is not bool:
                    raise ValueError("invalid boolean")
            elif kind == "enum":
                values = value if isinstance(value, list) else [value]
                if name in self.hierarchy and isinstance(value, list):
                    raise ValueError("hierarchy must be scalar")
                if not values or any(not isinstance(v, str) or v not in entry["values"] for v in values):
                    raise ValueError("invalid enum")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError("invalid string")
        if result.category == "רכב":
            result.params = {VEHICLE_KEYS.get(k, k): v for k, v in params.items()}
        return result
