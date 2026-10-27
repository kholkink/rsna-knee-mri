from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.constants import LABELS, resolve_label_columns
from rsna_knee.reports.labels import LabelMapConfig, build_label_table, gold_table, map_extraction
from rsna_knee.reports.rules import rule_extract
from rsna_knee.reports.schema import Finding, empty_extraction, json_schema


def test_resolve_label_columns_variants():
    cols = ["StudyInstanceUID", "ACL", "MCL", "Medial Meniscus", "Lateral_Meniscus", "Medial OA", "Lateral_OA",
            "Patellofemoral OA", "Effusion", "Synovitis", "Baker's Cyst", "Contusion", "Fracture"]
    m = resolve_label_columns(cols)
    assert set(m) == set(LABELS)
    assert m["bakers_cyst"] == "Baker's Cyst" and m["patellofemoral_oa"] == "Patellofemoral OA"


def test_schema_inlined_and_strict():
    s = json_schema()
    assert "$defs" not in s and s["additionalProperties"] is False
    assert s["properties"]["acl"]["properties"]["status"]["enum"] == ["present", "absent", "equivocal", "not_mentioned"]


def test_map_extraction_and_gold_override():
    ext = empty_extraction("en")
    ext.acl = Finding(status="present", severity="severe", detail="complete tear", evidence="complete ACL tear")
    ext.effusion = Finding(status="present", severity="mild", detail="trace", evidence="trace effusion")
    ext.mcl = Finding(status="absent", severity="unknown", detail="", evidence="MCL intact")
    ext.fracture = Finding(status="equivocal", severity="unknown", detail="healed", evidence="old fracture")
    cfg = LabelMapConfig(min_severity={"effusion": "moderate"})
    t, w = map_extraction(ext, cfg)
    assert t[LABELS.index("acl")] == 1.0 and t[LABELS.index("mcl")] == 0.0
    assert t[LABELS.index("effusion")] == cfg.below_value
    assert t[LABELS.index("fracture")] == 0.5 and w[LABELS.index("fracture")] == cfg.w_equivocal
    assert t[LABELS.index("medial_oa")] == cfg.not_mentioned_by_label["medial_oa"]

    train = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "Report": ["x", "y"], "ACL": [0.0, np.nan], "MCL": [1.0, np.nan],
                          **{c: [np.nan, np.nan] for c in ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
                                                            "Patellofemoral OA", "Effusion", "Synovitis", "Bakers Cyst",
                                                            "Contusion", "Fracture"]}})
    assert len(gold_table(train)) == 1
    table = build_label_table(train, {"a": ext, "b": ext}, cfg)
    ra, rb = table.set_index("study_uid").loc["a"], table.set_index("study_uid").loc["b"]
    assert ra["y_acl"] == 0.0 and ra["w_acl"] == cfg.w_gold and ra["is_gold"]
    assert rb["y_acl"] == 1.0 and rb["w_acl"] == 1.0 and not rb["is_gold"]


def test_rules_multilingual():
    en = rule_extract("Complete tear of the ACL. MCL is intact. No joint effusion. Small Baker's cyst. Menisci are normal.")
    assert en["acl"] == "positive" and en["mcl"] == "negative" and en["effusion"] == "negative"
    assert en["bakers_cyst"] == "positive"
    es = rule_extract("Rotura del ligamento cruzado anterior. Menisco medial sin alteraciones. Derrame articular moderado.")
    assert es["acl"] == "positive" and es["medial_meniscus"] == "negative" and es["effusion"] == "positive"
    ru = rule_extract("Разрыв передней крестообразной связки. Медиальный мениск не изменен. Выпот в полости сустава.")
    assert ru["acl"] == "positive" and ru["medial_meniscus"] == "negative" and ru["effusion"] == "positive"
    de = rule_extract("Kein Nachweis einer Fraktur. Knochenmarködem am lateralen Femurkondylus.")
    assert de["fracture"] == "negative" and de["contusion"] == "positive"
    assert rule_extract("Possible radial tear of the lateral meniscus.")["lateral_meniscus"] == "equivocal"


def test_example_json_and_openai_compat_prompt():
    from rsna_knee.reports.extract_llm import OpenAICompatBackend, parse_extraction
    from rsna_knee.reports.schema import example_extraction_json

    ext = parse_extraction("```json\n" + example_extraction_json() + "\n```")
    assert ext.acl.status == "present" and ext.language == "es"
    be = OpenAICompatBackend(model="m", base_url="http://localhost:1/v1", api_key="x", use_json_schema=False)
    sp = be.system_prompt()
    assert "json" in sp and '"medial_meniscus"' in sp


def test_pseudo_label_blend_keeps_gold_and_uncovered():
    import numpy as np
    import pandas as pd

    from rsna_knee.constants import LABELS
    from rsna_knee.pseudo import blend_pseudo_labels, collect_oof

    table = pd.DataFrame({"study_uid": ["a", "b", "g", "c"], "src": "llm", "is_gold": [False, False, True, False]})
    for k in LABELS:
        table[f"y_{k}"], table[f"w_{k}"] = [0.0, 1.0, 1.0, 0.2], 1.0
    oof1 = pd.DataFrame({"study_uid": ["a", "g"], "is_gold": [False, True], **{f"p_{k}": [0.8, 0.0] for k in LABELS}})
    oof2 = pd.DataFrame({"study_uid": ["a", "b"], "is_gold": [False, False], **{f"p_{k}": [0.4, 0.5] for k in LABELS}})
    oof = collect_oof([oof1, oof2])
    assert set(oof.index) == {"a", "b"} and oof.loc["a", "n_oof"] == 2
    new = blend_pseudo_labels(table, oof, 0.5).set_index("study_uid")
    assert np.isclose(new.loc["a", "y_acl"], 0.3)  # 0.5 * 0 + 0.5 * mean(0.8, 0.4)
    assert np.isclose(new.loc["b", "y_acl"], 0.75)
    assert new.loc["g", "y_acl"] == 1.0 and new.loc["g", "src"] == "llm"  # expert targets untouched
    assert np.isclose(new.loc["c", "y_acl"], 0.2) and new.loc["a", "src"] == "llm+oof"
