"""Value-level taxonomy rules for JSON assembly (deterministic; no language understanding).

- sector from sub-category: יד_שנייה/תתי_קטגוריות value -> its parent סקטורים (unambiguous in the taxonomy)
- manufacturer from model: רכב/דגמים value -> the יצרנים that lists it (when exactly one does)
- model from sub-model: רכב/תתי_דגמים value -> its דגמים (when exactly one model lists it)

A derived value fills a requested field the extractor left empty, or replaces one that contradicts the taxonomy.
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import Extraction


class TaxonomyRules:
    def __init__(self, taxonomy: dict):
        cats = taxonomy.get("קטגוריות", {})
        self.sub_to_sector = {}
        for sector, body in (cats.get("יד_שנייה", {}).get("סקטורים") or {}).items():
            for sub in (body.get("תתי_קטגוריות") or {}):
                self.sub_to_sector.setdefault(sub, set()).add(sector)
        self.model_to_maker, self.sub_to_model = {}, {}
        for maker, info in (cats.get("רכב", {}).get("יצרנים") or {}).items():
            for model in info.get("דגמים", []) or []:
                self.model_to_maker.setdefault(model, set()).add(maker)
            for model, subs in (info.get("תתי_דגמים") or {}).items():
                for s in subs:
                    self.sub_to_model.setdefault(s, set()).add(model)

    @classmethod
    def from_artifacts(cls, artifacts: Path) -> "TaxonomyRules":
        p = Path(artifacts) / "taxonomy_v2.json"
        return cls(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})

    def _derive(self, out: dict[str, Extraction], child: str, parent: str, table: dict):
        c, p = out.get(child), out.get(parent)
        if p is None or c is None or not c.found or not isinstance(c.value, str):
            return
        options = table.get(c.value, set())
        if len(options) != 1:
            return
        derived = next(iter(options))
        if not p.found or p.value != derived:
            out[parent] = Extraction(parent, derived, None, c.start, c.end, c.text, "taxonomy", c.score)

    def apply(self, domain: str | None, out: dict[str, Extraction]):
        if domain == "יד_שנייה":
            self._derive(out, "יד_שנייה/תתי_קטגוריות", "יד_שנייה/סקטורים", self.sub_to_sector)
        elif domain == "רכב":
            self._derive(out, "רכב/תתי_דגמים", "רכב/דגמים", self.sub_to_model)
            self._derive(out, "רכב/דגמים", "רכב/יצרנים", self.model_to_maker)
