"""LLM backends for structured report extraction.

* :class:`AnthropicBackend` - official ``anthropic`` SDK, Claude Opus 5, structured outputs
  (``output_config.format`` JSON schema), prompt-cached system prompt, optional Message Batches API
  (50 % cheaper, no rate-limit juggling - the right choice for the ~4.4k training reports).
* :class:`OpenAICompatBackend` - any OpenAI-compatible endpoint: a local vLLM server hosting an
  open-weight multilingual model (Qwen3 etc.), or DeepSeek. Use this if the competition rules forbid
  sending report text to a third-party API - a local model keeps the data on your machine.

Results are appended to a JSONL file as they arrive so a crashed run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pydantic import ValidationError
from tqdm import tqdm

from ..constants import ID_COL
from .prompts import SYSTEM_PROMPT, build_user_message
from .schema import ReportExtraction, example_extraction_json, json_schema

log = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Non-retryable failure (refusal, unparsable output)."""


class Backend(Protocol):
    name: str
    model: str

    def extract(self, report: str, patient_sex: str | None = None) -> ReportExtraction: ...


def parse_extraction(text: str) -> ReportExtraction:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return ReportExtraction.model_validate_json(t)
    except ValidationError as e:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return ReportExtraction.model_validate_json(t[start:end + 1])
            except ValidationError:
                pass
        raise ExtractionError(f"invalid structured output: {str(e)[:300]}") from e


def custom_id_for(study_uid: str) -> str:
    """Batch custom_ids must be short and alphanumeric; StudyInstanceUIDs are neither."""
    return "s" + hashlib.sha1(study_uid.encode()).hexdigest()[:40]


# ------------------------------------------------------------------------------- Anthropic (Claude)
@dataclass
class AnthropicBackend:
    model: str = "claude-opus-5"
    effort: str = "medium"       # low | medium | high - extraction is not reasoning-heavy
    max_tokens: int = 8000
    name: str = "anthropic"
    schema: dict[str, Any] = field(default_factory=json_schema, repr=False)

    def __post_init__(self) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # credentials from ANTHROPIC_API_KEY / `ant auth login`

    def request_params(self, report: str, patient_sex: str | None = None) -> dict[str, Any]:
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": build_user_message(report, patient_sex)}],
            output_config={"format": {"type": "json_schema", "schema": self.schema}, "effort": self.effort},
        )

    @staticmethod
    def _message_to_extraction(msg: Any) -> ReportExtraction:
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise ExtractionError(f"refusal ({cat})")
        if msg.stop_reason == "max_tokens":
            raise ExtractionError("max_tokens reached")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return parse_extraction(text)

    def extract(self, report: str, patient_sex: str | None = None) -> ReportExtraction:
        msg = self.client.messages.create(**self.request_params(report, patient_sex))
        return self._message_to_extraction(msg)

    # ---- Message Batches API -------------------------------------------------------------
    def submit_batch(self, items: list[tuple[str, str, str | None]]) -> str:
        """items: (study_uid, report, patient_sex) -> batch id."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        reqs = [
            Request(custom_id=custom_id_for(uid), params=MessageCreateParamsNonStreaming(**self.request_params(rep, sex)))
            for uid, rep, sex in items
        ]
        batch = self.client.messages.batches.create(requests=reqs)
        log.info("submitted batch %s with %d requests", batch.id, len(reqs))
        return batch.id

    def wait_batch(self, batch_id: str, poll_s: int = 60) -> None:
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                log.info("batch %s ended: %s", batch_id, b.request_counts)
                return
            log.info("batch %s: %s", batch_id, b.request_counts)
            time.sleep(poll_s)

    def collect_batch(self, batch_id: str) -> dict[str, ReportExtraction | Exception]:
        out: dict[str, ReportExtraction | Exception] = {}
        for r in self.client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                try:
                    out[r.custom_id] = self._message_to_extraction(r.result.message)
                except Exception as e:  # noqa: BLE001
                    out[r.custom_id] = e
            else:
                out[r.custom_id] = ExtractionError(f"batch result {r.result.type}")
        return out


# ------------------------------------------------------------------------ OpenAI-compatible (vLLM, DeepSeek)
@dataclass
class OpenAICompatBackend:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = 4000
    use_json_schema: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 600.0
    name: str = "openai_compat"
    schema: dict[str, Any] = field(default_factory=json_schema, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                             timeout=self.timeout_s, max_retries=3)
        self._usage_lock = threading.Lock()
        self.usage: dict[str, int] = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                      "reasoning_tokens": 0, "cache_hit_tokens": 0}

    def system_prompt(self) -> str:
        # never paste the JSON schema here (12x inlined Finding = ~4k tokens); a compact example is enough,
        # structure is enforced by response_format when the server supports json_schema
        return (SYSTEM_PROMPT + "\n\nReturn one json object with exactly these keys and nesting. The values below are "
                "only an example of the format, not of any real report:\n" + example_extraction_json())

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        with self._usage_lock:
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage["reasoning_tokens"] += int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
            self.usage["cache_hit_tokens"] += int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)

    def extract(self, report: str, patient_sex: str | None = None) -> ReportExtraction:
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": build_user_message(report, patient_sex)}]
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, max_tokens=self.max_tokens)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.use_json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "report_extraction", "schema": json_schema(compact=True), "strict": True}}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # server without json_schema support -> degrade once, permanently
            msg = str(e).lower()
            if self.use_json_schema and ("response_format" in msg or "json_schema" in msg):
                log.warning("json_schema response_format rejected (%s); falling back to json_object", e)
                self.use_json_schema = False
                return self.extract(report, patient_sex)
            raise
        self._record_usage(getattr(resp, "usage", None))
        choice = resp.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason == "length":
            raise ExtractionError("max_tokens reached")
        if not text.strip():
            raise ExtractionError("empty content")
        return parse_extraction(text)


@dataclass
class DeepSeekBackend(OpenAICompatBackend):
    """DeepSeek API (OpenAI-compatible). JSON Output mode only supports ``json_object``."""

    model: str = "deepseek-flash"
    base_url: str | None = "https://api.deepseek.com"
    temperature: float | None = None
    max_tokens: int = 16000
    use_json_schema: bool = False
    name: str = "deepseek"
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (put it into .env in the repo root)")
        if self.reasoning_effort:
            self.extra_body = {**self.extra_body, "reasoning_effort": self.reasoning_effort}
        super().__post_init__()


# ------------------------------------------------------------------------------------- runner
def load_extractions(path: str | Path) -> dict[str, ReportExtraction]:
    out: dict[str, ReportExtraction] = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("extraction"):
                try:
                    out[rec["study_uid"]] = ReportExtraction.model_validate(rec["extraction"])
                except ValidationError:
                    pass
    return out


def _append_jsonl(path: Path, rec: dict[str, Any], lock: threading.Lock) -> None:
    with lock, open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_extraction(
    df: pd.DataFrame,
    backend: Backend,
    out_path: str | Path,
    report_col: str = "Report",
    id_col: str = ID_COL,
    sex_col: str | None = "PatientSex",
    concurrency: int = 4,
    limit: int | None = None,
    max_retries: int = 4,
    log_every: int = 50,
    progress: bool = True,
) -> dict[str, ReportExtraction]:
    """Extract every report not already in ``out_path`` (resumable), ``concurrency`` requests in flight."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_extractions(out_path)
    todo = df[~df[id_col].isin(done)]
    todo = todo[todo[report_col].notna() & (todo[report_col].astype(str).str.strip() != "")]
    if limit:
        todo = todo.head(limit)
    lock = threading.Lock()

    def work(row: pd.Series) -> tuple[str, ReportExtraction | None, str]:
        uid = str(row[id_col])
        sex = str(row[sex_col]) if sex_col and sex_col in row and pd.notna(row[sex_col]) else None
        delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                return uid, backend.extract(str(row[report_col]), sex), ""
            except ExtractionError as e:
                if attempt >= 1:
                    return uid, None, str(e)
            except Exception as e:  # rate limits / transient network - SDK already retried
                if getattr(e, "status_code", None) in (400, 401, 403, 404, 413, 422):
                    return uid, None, f"{type(e).__name__}: {str(e)[:300]}"  # request itself is wrong: do not retry
                if attempt >= max_retries:
                    return uid, None, f"{type(e).__name__}: {e}"
                log.warning("retry %s (%s)", uid[:24], e)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        return uid, None, "unreachable"

    results = dict(done)
    t_start, n_done, n_err = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, row) for _, row in todo.iterrows()]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"extract[{backend.name}]", disable=not progress):
            uid, ext, err = fut.result()
            n_done += 1
            n_err += ext is None
            if n_done % log_every == 0 or n_done == len(futs):  # newline-terminated: Kaggle logs swallow tqdm's \r updates
                el = time.time() - t_start
                print(f"[extract] {n_done}/{len(futs)} done, {n_err} errors, {el / n_done:.1f}s/report, "
                      f"eta {el / n_done * (len(futs) - n_done) / 60:.0f} min", flush=True)
            rec = {"study_uid": uid, "backend": backend.name, "model": backend.model,
                   "extraction": ext.model_dump() if ext else None, "error": err}
            _append_jsonl(out_path, rec, lock)
            if ext is not None:
                results[uid] = ext
    return results


def run_batch_extraction(
    df: pd.DataFrame,
    backend: AnthropicBackend,
    out_path: str | Path,
    report_col: str = "Report",
    id_col: str = ID_COL,
    sex_col: str | None = "PatientSex",
    chunk: int = 1000,
    poll_s: int = 60,
) -> dict[str, ReportExtraction]:
    """Same as :func:`run_extraction` but through the Message Batches API (async, 50 % cost)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".batches.json")
    state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    done = load_extractions(out_path)
    todo = df[~df[id_col].isin(done)]
    todo = todo[todo[report_col].notna() & (todo[report_col].astype(str).str.strip() != "")]
    pending_ids = {b["id"] for b in state["batches"] if not b.get("collected")}
    already_submitted = {uid for b in state["batches"] for uid in b["uids"]} if pending_ids else set()
    todo = todo[~todo[id_col].isin(already_submitted)]
    items = [(str(r[id_col]), str(r[report_col]), (str(r[sex_col]) if sex_col and sex_col in r and pd.notna(r[sex_col]) else None))
             for _, r in todo.iterrows()]
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        bid = backend.submit_batch(part)
        state["batches"].append({"id": bid, "uids": [u for u, _, _ in part], "collected": False})
        state_path.write_text(json.dumps(state))
    lock = threading.Lock()
    results = dict(done)
    for b in state["batches"]:
        if b.get("collected"):
            continue
        backend.wait_batch(b["id"], poll_s)
        got = backend.collect_batch(b["id"])
        by_cid = {custom_id_for(u): u for u in b["uids"]}
        for cid, res in got.items():
            uid = by_cid.get(cid, cid)
            ok = isinstance(res, ReportExtraction)
            _append_jsonl(out_path, {"study_uid": uid, "backend": backend.name, "model": backend.model,
                                     "extraction": res.model_dump() if ok else None,
                                     "error": "" if ok else str(res)}, lock)
            if ok:
                results[uid] = res
        b["collected"] = True
        state_path.write_text(json.dumps(state))
    return results
