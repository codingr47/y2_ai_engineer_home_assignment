"""Manual root-cause labels for root_cause_sample.jsonl (one reviewer's judgment - hypotheses, not measurements).

    python training/experiment_1/experiment_analysis/annotate_sample.py

Writes root_cause_annotations.jsonl (sample + labels) and root_cause_summary.json (tallies).

Cause codes
    TWIN                 point/range twin fields (מחיר vs טווח_מחיר, קומה vs טווח_קומה, ...) - same wording, either label
    LABEL_INCONSISTENT   gold contradicts how the same cue is labelled elsewhere (e.g. 'באזור <city>' -> עיר vs אזור)
    TAXONOMY_OVERLAP     two distinct fields with overlapping meaning (אזור/עיר, יד/מספר_בעלים_קודמים, אבזור/מערכות_בטיחות)
    TAXONOMY_CONDITIONAL field choice depends on another field's value (מותג vs מותגים list, אחסון_GB vs נפח_אחסון)
    RULE_IMPLIED         field implied by a labelling rule rather than words (מגרש -> מגרשים, brand -> sector)
    ENTITY_KNOWLEDGE     needs a gazetteer / world knowledge (street, neighbourhood, transliterated brand, מקבוק -> Apple)
    TYPO_MORPH           typo, merged tokens, prefix morphology or mixed script hides the cue
    DILUTION             long / noisy query; the attribute is a small part of one pooled vector
    NEGATION             negated attribute ('בלי ריהוט') scored low
    THR_NEAR             score within 25% of the field threshold (threshold choice decides the outcome)
    THR_SETTING          the field's tuned threshold is extreme (≥0.5 or ≤0.2) and is what flips the decision
    PRIOR_FP             field predicted without textual evidence - co-occurrence prior of the domain
    LEXICAL_COLLISION    a word of the query collides with a field cue (רחובות ~ רחוב, יד ~ ידנית)
    SPURIOUS             FP with no visible lexical or prior explanation
    SPARSE               rare field value / item in train
    EMBEDDING_MISS       explicit, frequent cue in a short query, still scored far below threshold
    DOMAIN               domain head wrong; fields masked away (field_head_right = unmasked score was above threshold)
    CONTROL              correct prediction (control)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# case -> (primary, [secondary], note)
A = {
    1: ("THR_NEAR", ["DILUTION"], "מצב 0.36 vs thr 0.475; condition buried in a long rant"),
    2: ("LEXICAL_COLLISION", ["TYPO_MORPH"], "'ברחובות' contains 'רחוב' -> FP רחוב, city missed; merged 'שיהיהמעלית'"),
    3: ("SPARSE", ["DILUTION"], "rare item 'מרתף'; area as number words 'מארבעים'"),
    4: ("NEGATION", ["THR_SETTING"], "'בלי ריהוט' p=0.09; ריהוט threshold 0.75 (recall 0.39, precision 1.0)"),
    5: ("TAXONOMY_CONDITIONAL", [], "off-list brand -> מותג; model cannot know list membership"),
    6: ("TYPO_MORPH", ["TAXONOMY_OVERLAP"], "'סמס.ונג'; 'במרכז' region scored as עיר"),
    7: ("DILUTION", ["THR_NEAR"], "5 attributes + 'אני בן 70 07' noise; מצב near threshold"),
    8: ("ENTITY_KNOWLEDGE", [], "neighbourhood name 'כפר עקב'"),
    9: ("DILUTION", [], "city in an electronics query with joke tail"),
    10: ("TYPO_MORPH", [], "'שמורה' (feminine) + 'דירצ' typo; מצב_נכס p=0.01"),
    11: ("TWIN", [], "'4,100,000 ומעלה' gold טווח_מחיר, model מחיר (p=0.89)"),
    12: ("TYPO_MORPH", [], "merged 'בעלותבחכירה'"),
    13: ("DILUTION", ["SPARSE"], "long complaint; rare value גג/טיפוס"),
    14: ("TYPO_MORPH", ["DILUTION"], "'כשור' typo in keyword list; 3 fields missed"),
    15: ("SPARSE", ["TAXONOMY_OVERLAP"], "car leasing (בעלות: השכרה) rare; 'להשכרה' is a real-estate deal cue"),
    16: ("TWIN", [], "'שלא עבר 220K' gold טווח_ק״מ, model ק״מ (p=0.95)"),
    17: ("TWIN", ["LABEL_INCONSISTENT"], "'עד ...' gold טווח_מחיר; 'פרטי' gold בעלות (elsewhere סוגי_רכב)"),
    18: ("DILUTION", ["TWIN"], "price cue inside a 40-word chatty query"),
    19: ("THR_NEAR", [], "סוג_דלק 0.286 vs 0.325 ('פלאג-אין')"),
    20: ("EMBEDDING_MISS", ["TYPO_MORPH"], "'מקבלן' p=0.02 in a short query"),
    21: ("TWIN", ["PRIOR_FP"], "'לא מעל קומה 9' gold טווח_קומה, model קומה; FP סוגי_נכס"),
    22: ("TYPO_MORPH", ["RULE_IMPLIED"], "maker typo 'שברלוט'; kNN 1.0 yet LR 0.12"),
    23: ("TWIN", ["TAXONOMY_OVERLAP"], "'ממאה חמישים... ומעלה' gold טווח_מ״ר, model מ״ר_בנוי"),
    24: ("TYPO_MORPH", [], "'שיאומיי'"),
    25: ("DILUTION", ["NEGATION"], "long query, 3 misses incl. negated מחסן"),
    26: ("LABEL_INCONSISTENT", ["TWIN"], "'תקרה ...' (max) gold מחיר; same pattern elsewhere is טווח_מחיר - model chose טווח_מחיר"),
    27: ("TWIN", [], "range of rooms gold טווח_חדרים, model מס׳_חדרים (p=0.92)"),
    28: ("TYPO_MORPH", ["LABEL_INCONSISTENT"], "'מהשכ רה'; range price gold מחיר (cf. case 27: range -> טווח_)"),
    29: ("RULE_IMPLIED", [], "'מגרש' implies מצבי_עסקה=מגרשים by rule; p=0.05"),
    30: ("EMBEDDING_MISS", ["NEGATION"], "explicit 'קרוב לבתי ספר' p=0.003 next to 'בלי מיזוג'"),
    31: ("EMBEDDING_MISS", [], "'מ 2,000 ש\"ח ומעלה' p=0.23 (0.62·thr)"),
    32: ("TYPO_MORPH", [], "mixed-script slang 'חלאס MG HS אוטומטי'"),
    33: ("DILUTION", ["THR_SETTING"], "long polite query; סוגי_רהיט threshold 0.55"),
    34: ("TAXONOMY_OVERLAP", [], "adaptive cruise (מערכות_בטיחות) vs אבזור"),
    35: ("ENTITY_KNOWLEDGE", ["DILUTION"], "street 'לנדאו'"),
    36: ("TYPO_MORPH", [], "'וכסוף!' prefix + punctuation; צבע p=0.03"),
    37: ("RULE_IMPLIED", ["THR_NEAR"], "sector derived from brand 'שיאומי'; 0.20 vs 0.275"),
    38: ("TYPO_MORPH", [], "'מ.גרש' + self-correction '1150 ובעצם 1550'"),
    39: ("EMBEDDING_MISS", ["TYPO_MORPH"], "'לכל היותר יד אחת' p=0.06; Latin 'Audi' -> FP דגמים"),
    40: ("ENTITY_KNOWLEDGE", ["TYPO_MORPH"], "transliterated 'בי וי די סיל'"),
    41: ("PRIOR_FP", ["THR_NEAR"], "מותגים predicted, no brand in query"),
    42: ("PRIOR_FP", [], "מצבי_עסקה 0.84 with no deal word"),
    43: ("PRIOR_FP", ["THR_NEAR"], "מותגים 0.73, no brand; מצב 0.42 vs 0.475"),
    44: ("SPURIOUS", [], "בעלות 0.84 for 'BYD + שנת ייצור'"),
    45: ("SPURIOUS", ["LEXICAL_COLLISION"], "בעלות from 'מישהו מוכר'?"),
    46: ("PRIOR_FP", [], "חומר 0.89 for a furniture query with no material"),
    47: ("TAXONOMY_OVERLAP", [], "'בשרון' region gold אזור; model עיר"),
    48: ("THR_NEAR", [], "FP סוגי_נכס 0.333 vs 0.325"),
    49: ("TWIN", [], "'קומה בין 3 עד 10' gold טווח_קומה, model קומה"),
    50: ("SPURIOUS", [], "טעינה_מהירה 0.81 for 'יד שלישית או רביעית'"),
    51: ("TAXONOMY_CONDITIONAL", ["LABEL_INCONSISTENT", "ENTITY_KNOWLEDGE"],
         "laptop storage is אחסון_GB (phones נפח_אחסון); מקבוק->Apple; 'באזור חיפה' gold אזור"),
    52: ("TWIN", [], "'עד ...' gold טווח_מחיר, model מחיר (p=0.81)"),
    53: ("PRIOR_FP", [], "סוג 0.94 for 'בקבוקים ומוצצים' - possibly a label omission"),
    54: ("PRIOR_FP", [], "sector + sub-category predicted for 'משהו מיד שניה'"),
    55: ("TAXONOMY_OVERLAP", [], "'באזור השרון' gold אזור, model עיר 0.90"),
    56: ("TYPO_MORPH", [], "'חמשלי' -> סוג_דלק p=0.015"),
    57: ("TAXONOMY_OVERLAP", [], "'2-3 בעלים קודמים' -> FP יד"),
    58: ("TWIN", ["TYPO_MORPH"], "'מ-1 מליון ומעלה' gold טווח_מחיר; 'dים'"),
    59: ("THR_SETTING", [], "מצבי_עסקה threshold 0.2, p=0.27 with no deal word"),
    60: ("SPURIOUS", [], "טווח_חשמלי_ק״מ from horsepower numbers"),
    61: ("ENTITY_KNOWLEDGE", [], "street 'המואבים'; FP עיר"),
    62: ("LEXICAL_COLLISION", [], "generic 'רכב' -> סוגי_רכב 0.69"),
    63: ("RULE_IMPLIED", [], "'שטח מגרש' (a measure) -> FP מגרשים; the rule only applies to מגרש as the item - FP defensible"),
    64: ("SPURIOUS", [], "צבע 0.49 with no colour"),
    65: ("THR_NEAR", ["SPURIOUS"], "צבע 0.267 vs 0.275; FP עיר"),
    66: ("TWIN", [], "range gold טווח_קומה; model added קומה too"),
    67: ("LABEL_INCONSISTENT", ["TAXONOMY_OVERLAP"], "'באזור באר שבע' gold עיר (case 51: 'באזור חיפה' gold אזור)"),
    68: ("SPURIOUS", [], "קרבה 0.81 for street + area query"),
    69: ("LEXICAL_COLLISION", ["TAXONOMY_OVERLAP"], "'יד 2' -> תיבת_הילוכים 0.96 (ידנית) and מספר_בעלים_קודמים"),
    70: ("PRIOR_FP", [], "סוגי_נכס 0.88, no property type"),
    71: ("DOMAIN", ["TYPO_MORPH"], "'ר כב משנת 2013' -> יד_שנייה (שנת_ייצור)"),
    72: ("DOMAIN", ["TYPO_MORPH"], "'אפנוע' typo -> יד_שנייה"),
    73: ("DOMAIN", ["DILUTION"], "long slang baby-item query -> נדל״ן (conf 0.56)"),
    74: ("DOMAIN", [], "self-correcting 'חניה 2 רגע בעצם חניה 1' -> רכב"),
    75: ("DOMAIN", [], "car query with site-question phrasing -> none"),
    76: ("DOMAIN", [], "two-word query -> none"),
    77: ("DOMAIN", [], "question register -> יד_שנייה"),
    78: ("DOMAIN", ["DILUTION", "TYPO_MORPH"], "-> נדל״ן at conf 0.37"),
    79: ("DOMAIN", [], "negative near-miss (swap offer) -> יד_שנייה"),
    80: ("DOMAIN", [], "short slang -> none"),
}
for c in range(81, 101):
    A[c] = ("CONTROL", [], "")


def main():
    sample = [json.loads(line) for line in open(HERE / "root_cause_sample.jsonl", encoding="utf-8")]
    assert len(sample) == len(A) == 100
    out, by_reason = [], {}
    for s in sample:
        primary, secondary, note = A[s["case"]]
        # measured cross-checks next to the judgment
        fn_ratios = [s["scores"][f]["p"] / s["scores"][f]["thr"] for f in s["false_negatives"]]
        s["near_miss_fn"] = any(r >= 0.75 for r in fn_ratios)
        s["field_head_right"] = (s["reason"] == "domain_error"
                                 and all(s["scores"][f]["p"] >= s["scores"][f]["thr"] for f in s["false_negatives"])
                                 and bool(s["false_negatives"]))
        s.update({"primary_cause": primary, "secondary_causes": secondary, "note": note})
        out.append(s)
        by_reason.setdefault(s["reason"], Counter())[primary] += 1
    with open(HERE / "root_cause_annotations.jsonl", "w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    errors = [s for s in out if s["primary_cause"] != "CONTROL"]
    summary = {
        "n_cases": len(out), "n_errors": len(errors),
        "primary": Counter(s["primary_cause"] for s in errors).most_common(),
        "any_mention": Counter(c for s in errors for c in [s["primary_cause"], *s["secondary_causes"]]).most_common(),
        "primary_by_reason": {k: v.most_common() for k, v in by_reason.items()},
        "near_miss_fn_cases": sum(s["near_miss_fn"] for s in errors),
        "domain_errors_with_field_head_right": sum(s["field_head_right"] for s in out),
        "note": "labels are one reviewer's judgment on a stratified sample (not a random sample); "
                "use for direction, not for population rates",
    }
    (HERE / "root_cause_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
