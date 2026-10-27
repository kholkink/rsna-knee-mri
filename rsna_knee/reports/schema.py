"""Structured-extraction schema shared by every LLM backend and by the label mapper."""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..constants import LABELS

Status = Literal["present", "absent", "equivocal", "not_mentioned"]
Severity = Literal["unknown", "mild", "moderate", "severe"]
Laterality = Literal["left", "right", "bilateral", "unknown"]


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status = Field(description=(
        "present = the report states the finding exists; absent = explicitly negated or structure called "
        "normal/intact; equivocal = hedged/possible/cannot exclude/borderline; not_mentioned = the report does "
        "not address this structure or finding at all"))
    severity: Severity = Field(description=(
        "Grade as stated or implied by the report: mild (small/trace/low-grade/partial-thickness/grade 1), "
        "moderate (moderate/partial tear/high-grade partial-thickness), severe (large/complete/full-thickness/"
        "grade 3/bucket-handle/root tear); unknown when no grade can be inferred or status is absent/not_mentioned"))
    detail: str = Field(description=(
        "Short English descriptor of subtype, e.g. 'complete tear', 'partial tear', 'mucoid degeneration', "
        "'root tear', 'intrasubstance signal', 'trace', 'subchondral edema', 'healed'. Empty string if none"))
    evidence: str = Field(description=(
        "Shortest verbatim quote (original language, max ~160 characters) supporting the status. "
        "Empty string when not_mentioned"))


class ReportExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO-639-1 code of the report language, e.g. 'en', 'es', 'pt', 'de'")
    laterality: Laterality = Field(description="Which knee the report describes")
    prior_surgery: bool = Field(description="True if prior knee surgery (ACL reconstruction, meniscectomy, etc.) is mentioned")
    acl: Finding
    mcl: Finding
    medial_meniscus: Finding
    lateral_meniscus: Finding
    medial_oa: Finding
    lateral_oa: Finding
    patellofemoral_oa: Finding
    effusion: Finding
    synovitis: Finding
    bakers_cyst: Finding
    contusion: Finding
    fracture: Finding
    other_findings: str = Field(description="Other notable findings not covered by the 12 labels, in English, brief")

    def findings(self) -> dict[str, Finding]:
        return {k: getattr(self, k) for k in LABELS}


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` -> ``$defs`` so backends that dislike references (some vLLM builds) accept it."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return walk(copy.deepcopy(defs[name]))
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _strip_docs(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_docs(v) for k, v in node.items()
                if not (k in ("description", "title") and isinstance(v, str))}
    if isinstance(node, list):
        return [_strip_docs(v) for v in node]
    return node


def json_schema(inline: bool = True, compact: bool = False) -> dict[str, Any]:
    """``compact`` drops descriptions/titles: the grammar for guided decoding needs structure only.

    (The Finding schema is inlined 12x, so the documented variant costs ~4k prompt tokens if pasted into a prompt.)
    """
    schema = ReportExtraction.model_json_schema()
    schema = _inline_refs(schema) if inline else schema
    return _strip_docs(schema) if compact else schema


def empty_extraction(language: str = "unknown") -> ReportExtraction:
    f = Finding(status="not_mentioned", severity="unknown", detail="", evidence="")
    return ReportExtraction(language=language, laterality="unknown", prior_surgery=False,
                            **{k: f for k in LABELS}, other_findings="")


def example_extraction_json() -> str:
    """Compact example object for backends that only support ``json_object`` (DeepSeek)."""
    ex = empty_extraction("es")
    ex.laterality = "right"
    ex.acl = Finding(status="present", severity="severe", detail="complete tear", evidence="Rotura completa del LCA")
    ex.mcl = Finding(status="absent", severity="unknown", detail="", evidence="LCM de morfología normal")
    ex.medial_meniscus = Finding(status="equivocal", severity="unknown", detail="intrasubstance signal",
                                 evidence="aumento de señal intrasustancia sin contacto articular")
    ex.effusion = Finding(status="present", severity="mild", detail="small", evidence="Derrame articular leve")
    ex.other_findings = "PCL intact"
    return ex.model_dump_json()
