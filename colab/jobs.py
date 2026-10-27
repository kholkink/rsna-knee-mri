#!/usr/bin/env python
"""Queue for the Colab workers (colab/worker.py): a private Kaggle dataset <user>/rsna-knee-jobs holding jobs.json.

    python colab/jobs.py add --worker A --name v14-nano-long --seeds 0 1 --args "--config configs/... --override ..."
    python colab/jobs.py disable --name v14-nano-long      # or: enable / rm
    python colab/jobs.py list                              # queue + which seeds are already on Kaggle
    python colab/jobs.py publish                           # workers see the change at their next poll (<= 5 min)
    python colab/jobs.py status                            # what each worker reported last

`args` are colab_train.py arguments without --name / --folds / --drive-root / --upload (the worker sets those).
A worker trains one seed, re-reads the queue, and takes the first job of its letter that still has seeds to train.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "colab"))
from worker import FORBIDDEN_ARGS, parse_done_seeds

QDIR = os.path.join(REPO, "build", "jobs")
QFILE = os.path.join(QDIR, "jobs.json")
KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")


def load() -> dict:
    if os.path.exists(QFILE):
        with open(QFILE) as f:
            return json.load(f)
    return {"jobs": []}


def save(q: dict) -> None:
    os.makedirs(QDIR, exist_ok=True)
    with open(QFILE, "w") as f:
        json.dump(q, f, indent=1)


def done_seeds(user: str, name: str) -> set[int]:
    r = subprocess.run([KAGGLE, "datasets", "files", f"{user}/rsna-knee-runs-{name}", "--csv", "--page-size", "200"], capture_output=True, text=True, check=False)
    return parse_done_seeds(r.stdout) if r.returncode == 0 else set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["add", "rm", "disable", "enable", "list", "publish", "status", "stop", "resume"])
    ap.add_argument("--name")
    ap.add_argument("--worker", default="A")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--args", default="", help="colab_train.py arguments as one string")
    ap.add_argument("--user", default="kirillkholkin")
    a = ap.parse_args()
    q = load()

    if a.cmd == "add":
        args = shlex.split(a.args)
        bad = [x for x in args if x.split("=")[0] in FORBIDDEN_ARGS]
        if bad or not a.name:
            raise SystemExit(f"need --name, and args must not contain {FORBIDDEN_ARGS} (got {bad})")
        q["jobs"] = [j for j in q["jobs"] if j["name"] != a.name] + [{"name": a.name, "worker": a.worker.upper(), "seeds": a.seeds, "args": args}]
        save(q)
        print(f"queued {a.name} for worker {a.worker.upper()}, seeds {a.seeds}")
    elif a.cmd in ("rm", "disable", "enable"):
        hit = [j for j in q["jobs"] if j["name"] == a.name]
        if not hit:
            raise SystemExit(f"no job {a.name}")
        if a.cmd == "rm":
            q["jobs"] = [j for j in q["jobs"] if j["name"] != a.name]
        else:
            hit[0]["disabled"] = a.cmd == "disable"
        save(q)
        print(f"{a.cmd}: {a.name}")
    elif a.cmd in ("stop", "resume"):  # stop: the worker releases its runtime at its next poll (between seeds)
        st = {w.upper() for w in q.get("stop_workers", [])}
        st = st | {a.worker.upper()} if a.cmd == "stop" else st - {a.worker.upper()}
        q["stop_workers"] = sorted(st)
        save(q)
        print(f"stop_workers = {q['stop_workers']} (publish to apply)")
    elif a.cmd == "list":
        for j in q["jobs"]:
            done = done_seeds(a.user, j["name"])
            flag = " (disabled)" if j.get("disabled") else ""
            print(f"{j['worker']}  {j['name']}{flag}: seeds {j['seeds']}, on Kaggle {sorted(done)}")
    elif a.cmd == "publish":
        save(q)
        meta = {"title": "rsna-knee-jobs", "id": f"{a.user}/rsna-knee-jobs", "licenses": [{"name": "CC0-1.0"}]}
        with open(os.path.join(QDIR, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)
        r = subprocess.run([KAGGLE, "datasets", "version", "-p", QDIR, "-m", "queue update", "-q"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            r = subprocess.run([KAGGLE, "datasets", "create", "-p", QDIR, "-q"], capture_output=True, text=True, check=False)
        print("published" if r.returncode == 0 else f"publish failed: {(r.stderr or r.stdout)[-300:]}")
    elif a.cmd == "status":
        for w in "abcdef":
            d = os.path.join(REPO, "build", "jobs_status", w)
            r = subprocess.run([KAGGLE, "datasets", "download", f"{a.user}/rsna-knee-colab-status-{w}", "-p", d, "--unzip", "-o", "-q"],
                               capture_output=True, text=True, check=False)
            p = os.path.join(d, "status.json")
            if r.returncode == 0 and os.path.exists(p):
                with open(p) as f:
                    print(json.load(f))


if __name__ == "__main__":
    main()
