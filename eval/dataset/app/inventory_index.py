"""Index over the field inventory + dimensions matrix used by the dataset editor.

The inventory (eval/dataset/inventory/field_inventory.json) is generated from
yad2_search_taxonomy.json. Each entry carries `conditions`: the hierarchy picks
(e.g. {"יצרנים": "טויוטה"}) under which the field applies. This module answers:

* which hierarchy levels are active for the current picks   -> active_hierarchy()
* which (non-hierarchy) fields are available for those picks -> available_fields()

Hierarchy keys are derived structurally: every selector entry plus every field
name that some other entry is conditioned on (e.g. "דגמים" drives "תתי_דגמים").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARAM_KINDS = {"field", "selector", "dependent_field"}  # "mapping" (typo maps) is not a param

# ASCII '"' between Hebrew letters is a stand-in for gershayim (U+05F4): נדל"ן -> נדל״ן
_HEB_QUOTE = re.compile(r'(?<=[א-ת])"(?=[א-ת])')


def normalize_hebrew(text: str) -> str:
    return _HEB_QUOTE.sub("״", text)


def conditions_met(conditions: dict, picks: dict) -> bool:
    return all(picks.get(k) == v for k, v in conditions.items())


@dataclass
class Dimension:
    name: str
    values: list[str]


@dataclass
class InventoryIndex:
    domains: list[str]
    entries: dict[str, list[dict]]            # domain -> param entries in inventory order
    hierarchy_keys: dict[str, set[str]]       # domain -> field names that drive the hierarchy
    dimensions: list[Dimension] = field(default_factory=list)

    # -- hierarchy ---------------------------------------------------------
    def active_hierarchy(self, domain: str, picks: dict) -> list[dict]:
        """Hierarchy levels whose conditions are satisfied by `picks`, in inventory order."""
        keys = self.hierarchy_keys.get(domain, set())
        return [e for e in self.entries.get(domain, [])
                if e["field_name"] in keys and conditions_met(e["conditions"], picks)]

    def hierarchy_entry(self, domain: str, name: str, picks: dict) -> dict | None:
        return next((e for e in self.active_hierarchy(domain, picks) if e["field_name"] == name), None)

    # -- fields ------------------------------------------------------------
    def available_fields(self, domain: str, picks: dict) -> dict[str, dict]:
        """Non-hierarchy fields available for `picks`, keyed by field name.

        When a field name appears more than once (e.g. general "מצב" and the
        sub-category-specific "מצב"), the most specific entry (most conditions) wins.
        """
        keys = self.hierarchy_keys.get(domain, set())
        out: dict[str, dict] = {}
        for e in self.entries.get(domain, []):
            name = e["field_name"]
            if name in keys or not conditions_met(e["conditions"], picks):
                continue
            if name not in out or len(e["conditions"]) > len(out[name]["conditions"]):
                out[name] = e
        return out

    def picks_from_params(self, domain: str, params: dict) -> dict:
        """Recover hierarchy picks from a params dict (in hierarchy order)."""
        picks: dict[str, str] = {}
        while True:
            added = False
            for e in self.active_hierarchy(domain, picks):
                name = e["field_name"]
                if name not in picks and isinstance(params.get(name), str):
                    picks[name] = params[name]
                    added = True
            if not added:
                return picks

    def dimension(self, name: str) -> Dimension | None:
        return next((d for d in self.dimensions if d.name == name), None)


def load_index(inventory_path: Path, dimensions_path: Path | None = None) -> InventoryIndex:
    inv = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    entries: dict[str, list[dict]] = {}
    for e in inv["entries"]:
        if e["kind"] in PARAM_KINDS:
            entries.setdefault(e["domain"], []).append(e)

    hierarchy_keys: dict[str, set[str]] = {}
    for domain, es in entries.items():
        keys = {e["field_name"] for e in es if e["kind"] == "selector"}
        for e in es:
            keys.update(e["conditions"])
        hierarchy_keys[domain] = keys

    dimensions: list[Dimension] = []
    if dimensions_path is not None:
        raw = json.loads(Path(dimensions_path).read_text(encoding="utf-8"))
        dimensions = [Dimension(d["dimension"], [normalize_hebrew(str(v)) for v in d["values_to_cover"]])
                      for d in raw]

    return InventoryIndex(domains=list(entries), entries=entries,
                          hierarchy_keys=hierarchy_keys, dimensions=dimensions)


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
