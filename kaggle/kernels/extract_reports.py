# ---- kernel body: report -> structured findings with an open-weight LLM served by vLLM INSIDE Kaggle ----
# Report text never leaves Kaggle. GPU kernel (T4 x2), internet on (pip + model download only).
import json, os, subprocess, sys, time, urllib.request

T0 = time.time()
P = {"attempts": [{"model": "Qwen/Qwen3-14B-AWQ", "tp": 2}, {"model": "Qwen/Qwen3-8B-AWQ", "tp": 2}],
     "max_model_len": 10240, "max_tokens": 1600, "concurrency": 48, "limit": 0, "thinking": False,
     "vllm_spec": "vllm", "startup_timeout_s": 2400, **KERNEL_PARAMS}
ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK = os.environ.get("RSNA_WORK_DIR", "/kaggle/working" if ON_KAGGLE else "work")
os.makedirs(WORK, exist_ok=True)

import pandas as pd
from rsna_knee.constants import ID_COL, LABELS, resolve_label_columns
from rsna_knee.paths import find_comp_root
from rsna_knee.reports.extract_llm import OpenAICompatBackend, load_extractions, run_extraction
from rsna_knee.reports.labels import LabelMapConfig, evaluate_mapping, gold_table

comp = find_comp_root()
train = pd.read_csv(os.path.join(comp, "train.csv"))
train[ID_COL] = train[ID_COL].astype(str)
gold_cols = list(resolve_label_columns(train.columns).values())
is_gold = train[gold_cols].notna().any(axis=1)
train = pd.concat([train[is_gold], train[~is_gold]], ignore_index=True)  # gold first -> early agreement table
if P["limit"]:
    train = train.head(P["limit"])
gold = gold_table(train)
print(f"{len(train)} reports, {len(gold)} gold")


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def start_server(model, tp, log_path):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", "extractor",
           "--tensor-parallel-size", str(tp), "--dtype", "half", "--max-model-len", str(P["max_model_len"]),
           "--gpu-memory-utilization", "0.90", "--max-num-seqs", "64", "--enforce-eager", "--port", "8000",
           "--enable-prefix-caching"]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    t = time.time()
    while time.time() - t < P["startup_timeout_s"]:
        if proc.poll() is not None:
            return None
        try:
            if urllib.request.urlopen("http://localhost:8000/health", timeout=5).status == 200:
                print(f"server up in {time.time() - t:.0f}s", flush=True)
                return proc
        except Exception:
            time.sleep(10)
    proc.kill()
    return None


def agreement(out_path, title):
    ext = load_extractions(out_path)
    if len(gold):
        ev = evaluate_mapping(ext, gold, LabelMapConfig())
        print(f"--- {title}: agreement with gold ({len(ext)} extractions) ---\n{ev.to_string(index=False)}\nmean AUC {ev['auc'].mean():.3f}", flush=True)
        ev.to_csv(os.path.join(WORK, "gold_agreement.csv"), index=False)


if ON_KAGGLE:
    sh([sys.executable, "-m", "pip", "install", "-q", P["vllm_spec"]], check=False)
    sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

done = False
for k, att in enumerate(P["attempts"]):
    log_path = os.path.join(WORK, f"vllm_{k}.log")
    proc = start_server(att["model"], att["tp"], log_path)
    if proc is None:
        print(f"vLLM failed for {att}; log tail:\n" + "".join(open(log_path).readlines()[-40:]), flush=True)
        continue
    out_path = os.path.join(WORK, "extractions.jsonl")
    extra = {} if P["thinking"] else {"chat_template_kwargs": {"enable_thinking": False}}
    be = OpenAICompatBackend(model="extractor", base_url="http://localhost:8000/v1", api_key="EMPTY", temperature=0.0,
                             max_tokens=P["max_tokens"], extra_body=extra)
    be.name = "vllm:" + att["model"]
    n_pilot = int(is_gold.sum()) if not P["limit"] else min(P["limit"], int(is_gold.sum()))
    if n_pilot:
        got = run_extraction(train.head(n_pilot), be, out_path, concurrency=P["concurrency"])
        agreement(out_path, "pilot")
        ok = sum(1 for u in train.head(n_pilot)[ID_COL] if u in got)
        print(f"pilot: {ok}/{n_pilot} ok; usage={be.usage}", flush=True)
        if ok < 0.8 * n_pilot:  # do not burn GPU quota on a broken setup
            errs = [json.loads(l).get("error") for l in open(out_path)][-3:]
            proc.terminate()
            raise SystemExit(f"pilot failed ({ok}/{n_pilot}); last errors: {errs}")
    res = run_extraction(train, be, out_path, concurrency=P["concurrency"])
    agreement(out_path, "final")
    print(f"{len(res)}/{len(train)} extracted with {att['model']} in {time.time() - T0:.0f}s; usage={be.usage}", flush=True)
    proc.terminate()
    done = len(res) >= 0.97 * len(train)
    if done:
        break
json.dump({"done": done, "seconds": round(time.time() - T0)}, open(os.path.join(WORK, "summary_extract.json"), "w"))
if not done:
    raise SystemExit("extraction incomplete - inspect vllm_*.log")
