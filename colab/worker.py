"""Colab worker: runs the training jobs queued in the private Kaggle dataset <user>/rsna-knee-jobs.

One cell per Colab session (after the Kaggle token is in place), the letter picks this session's jobs:

    %run /content/code/colab/worker.py --worker A

The worker never needs a notebook edit again: jobs are added, reordered or disabled from outside with
`python colab/jobs.py ...`, which publishes a new version of the jobs dataset. The loop:

  * polls jobs.json every few minutes and takes the first job of its letter that still has seeds to train;
  * a seed is done when fold<seed>/best.pt is in the dataset <user>/rsna-knee-runs-<name> (or was trained here);
  * before a job it refreshes the code dataset and restores the seeds already published, because every upload
    replaces the dataset version and must keep the earlier seeds;
  * after every seed it publishes a small status dataset <user>/rsna-knee-colab-status-<letter> (state + log tail);
  * when it stays idle for --idle-minutes (or hits --max-hours) it releases the runtime, so no units burn.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time

FORBIDDEN_ARGS = ("--name", "--folds", "--drive-root", "--upload")  # the worker sets these itself


def log(msg: str) -> None:
    print(f"[worker {dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def kaggle_bin() -> str:
    return shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")


def run_quiet(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, e.stdout or "", f"timeout after {timeout}s")
    except OSError as e:  # e.g. nvidia-smi does not exist on a CPU runtime
        return subprocess.CompletedProcess(cmd, 127, "", f"{type(e).__name__}: {e}")


def parse_done_seeds(files_csv: str) -> set[int]:
    """`kaggle datasets files --csv` output -> seeds k with fold<k>/best.pt present."""
    done = set()
    for row in csv.reader(io.StringIO(files_csv)):
        if not row:
            continue
        name = row[0].replace("\\", "/")
        parts = name.split("/")
        if len(parts) >= 2 and parts[-1] == "best.pt" and parts[-2].startswith("fold") and parts[-2][4:].isdigit():
            done.add(int(parts[-2][4:]))
    return done


def ckpt_slug(name: str, seed: int) -> str:
    """Kaggle dataset slug for the rolling checkpoint of one seed (slugs: lowercase, <= 50 characters)."""
    return f"rsna-knee-ckpt-{name}-s{seed}".lower()[:50]


def pick_job(jobs: list[dict], worker: str, done_of, skip: set[tuple[str, int]]) -> tuple[dict, list[int]] | None:
    """First enabled job of this worker with seeds left; `done_of(name)` -> set of finished seeds."""
    for j in jobs:
        if str(j.get("worker", "")).upper() != worker or j.get("disabled"):
            continue
        done = done_of(j["name"])
        missing = [int(s) for s in j.get("seeds", [0]) if int(s) not in done and (j["name"], int(s)) not in skip]
        if missing:
            return j, missing
    return None


def seed_command(job: dict, seed: int, work_root: str, skip_pip: bool) -> list[str]:
    args = [str(x) for x in job.get("args", [])]
    bad = [x for x in args if x.split("=")[0] in FORBIDDEN_ARGS]
    if bad:
        raise ValueError(f"job {job['name']}: arguments {bad} are set by the worker")
    cmd = [sys.executable, "colab/colab_train.py", "--name", job["name"], "--folds", str(seed), "--drive-root", work_root, *args, "--upload"]
    return cmd + (["--skip-pip"] if skip_pip else [])


class Worker:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.letter = a.worker.upper()
        self.k = kaggle_bin()
        self.trained: set[tuple[str, int]] = set()   # finished here (the dataset listing lags by minutes)
        self.failed: set[tuple[str, int]] = set()    # never retried in this session
        self.pip_done = False
        self.status_dir = f"/content/worker_status_{self.letter.lower()}"
        self.status_id = f"{a.user}/rsna-knee-colab-status-{self.letter.lower()}"

    # ------------------------------------------------------------------ kaggle helpers
    def fetch_queue(self) -> dict:
        d = "/content/jobs"
        shutil.rmtree(d, ignore_errors=True)
        r = run_quiet([self.k, "datasets", "download", f"{self.a.user}/{self.a.jobs_dataset}", "-p", d, "--unzip", "-o", "-q"])
        files = glob.glob(os.path.join(d, "**", "jobs.json"), recursive=True)
        if r.returncode != 0 or not files:
            log(f"cannot read the jobs dataset ({(r.stderr or r.stdout)[-200:].strip()})")
            return {}
        with open(files[0]) as f:
            return json.load(f)

    def fetch_jobs(self) -> list[dict]:
        d = "/content/jobs"
        shutil.rmtree(d, ignore_errors=True)
        r = run_quiet([self.k, "datasets", "download", f"{self.a.user}/{self.a.jobs_dataset}", "-p", d, "--unzip", "-o", "-q"])
        files = glob.glob(os.path.join(d, "**", "jobs.json"), recursive=True)
        if r.returncode != 0 or not files:
            log(f"cannot read the jobs dataset ({(r.stderr or r.stdout)[-200:].strip()})")
            return []
        with open(files[0]) as f:
            return json.load(f).get("jobs", [])

    def done_seeds(self, name: str) -> set[int]:
        r = run_quiet([self.k, "datasets", "files", f"{self.a.user}/rsna-knee-runs-{name}", "--csv", "--page-size", "200"], timeout=120)
        remote = parse_done_seeds(r.stdout) if r.returncode == 0 else set()
        return remote | {s for n, s in self.trained if n == name}

    def refresh_code(self) -> None:
        tmp = self.a.code_dir + ".new"
        shutil.rmtree(tmp, ignore_errors=True)
        r = run_quiet([self.k, "datasets", "download", f"{self.a.user}/rsna-knee-code", "-p", tmp, "--unzip", "-o", "-q"])
        if r.returncode == 0 and os.path.exists(os.path.join(tmp, "colab", "colab_train.py")):
            shutil.rmtree(self.a.code_dir, ignore_errors=True)
            os.replace(tmp, self.a.code_dir)
            log("code refreshed from the code dataset")
        else:
            log(f"code refresh failed, keeping the current copy ({(r.stderr or '')[-200:].strip()})")

    def restore_runs(self, name: str, done: set[int]) -> None:
        """Put already published seeds back under runs/<name>/ so the next upload keeps them."""
        dest = os.path.join(self.a.work_root, f"work-{name}", "runs", name)
        have = {int(p.rsplit("fold", 1)[1]) for p in glob.glob(os.path.join(dest, "fold*")) if p.rsplit("fold", 1)[1].isdigit()}
        if not done - have:
            return
        log(f"restoring published seeds {sorted(done - have)} of {name}")
        r = run_quiet([self.k, "datasets", "download", f"{self.a.user}/rsna-knee-runs-{name}", "-p", dest, "--unzip", "-o", "-q"], timeout=3600)
        if r.returncode != 0:
            log(f"restore failed: {(r.stderr or r.stdout)[-300:].strip()}")

    def report(self, state: str, job: str = "", seed: int | None = None, rc: int | None = None, tail: str = "") -> None:
        os.makedirs(self.status_dir, exist_ok=True)
        st = {"worker": self.letter, "time_utc": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}", "state": state, "job": job, "seed": seed,
              "rc": rc, "trained": sorted(f"{n}#{s}" for n, s in self.trained), "failed": sorted(f"{n}#{s}" for n, s in self.failed)}
        with open(os.path.join(self.status_dir, "status.json"), "w") as f:
            json.dump(st, f, indent=1)
        with open(os.path.join(self.status_dir, "log_tail.txt"), "w") as f:
            f.write(tail[-20000:])
        meta = {"title": self.status_id.split("/")[1], "id": self.status_id, "licenses": [{"name": "CC0-1.0"}]}
        with open(os.path.join(self.status_dir, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)
        r = run_quiet([self.k, "datasets", "version", "-p", self.status_dir, "-m", f"{state} {job} {seed}", "-q"], timeout=300)
        if r.returncode != 0:
            r = run_quiet([self.k, "datasets", "create", "-p", self.status_dir, "-q"], timeout=300)
        if r.returncode != 0:
            log(f"status upload failed: {(r.stderr or r.stdout)[-200:].strip()}")

    # ------------------------------------------------------------------ checkpoints (a dead runtime must not cost the seed)
    def fold_dir(self, name: str, seed: int) -> str:
        return os.path.join(self.a.work_root, f"work-{name}", "runs", name, f"fold{seed}")

    def ckpt_id(self, name: str, seed: int) -> str:
        return f"{self.a.user}/{ckpt_slug(name, seed)}"

    def restore_ckpt(self, name: str, seed: int) -> None:
        """last.pt + history of an interrupted seed from its checkpoint dataset -> training resumes at the next epoch."""
        fold = self.fold_dir(name, seed)
        if os.path.exists(os.path.join(fold, "last.pt")):
            return
        tmp = os.path.join(self.a.work_root, f"ckpt_restore_{self.letter.lower()}")
        shutil.rmtree(tmp, ignore_errors=True)
        r = run_quiet([self.k, "datasets", "download", self.ckpt_id(name, seed), "-p", tmp, "--unzip", "-o", "-q"], timeout=3600)
        if r.returncode != 0 or not os.path.exists(os.path.join(tmp, "last.pt")):
            return
        os.makedirs(fold, exist_ok=True)
        for f in os.listdir(tmp):
            if f != "dataset-metadata.json":
                shutil.move(os.path.join(tmp, f), os.path.join(fold, f))
        log(f"resuming {name} seed {seed} from its checkpoint dataset")

    def ckpt_uploader(self, name: str, seed: int, stop: threading.Event) -> None:
        fold, stage = self.fold_dir(name, seed), os.path.join(self.a.work_root, f"ckpt_stage_{self.letter.lower()}")
        slug = self.ckpt_id(name, seed)
        last_mtime = 0.0
        while not stop.wait(self.a.ckpt_minutes * 60):
            lp = os.path.join(fold, "last.pt")
            if not os.path.exists(lp) or os.path.getmtime(lp) == last_mtime:
                continue
            try:
                shutil.rmtree(stage, ignore_errors=True)
                os.makedirs(stage)
                m0 = os.path.getmtime(lp)
                shutil.copy2(lp, os.path.join(stage, "last.pt"))
                if os.path.getmtime(lp) != m0:  # copied while the trainer was writing: try at the next tick
                    continue
                for f in ("history.jsonl", "priors.json", "config.yaml"):
                    if os.path.exists(os.path.join(fold, f)):
                        shutil.copy2(os.path.join(fold, f), os.path.join(stage, f))
                with open(os.path.join(stage, "dataset-metadata.json"), "w") as fh:
                    json.dump({"title": slug.split("/")[1], "id": slug, "licenses": [{"name": "CC0-1.0"}]}, fh)
                r = run_quiet([self.k, "datasets", "version", "-p", stage, "-m", "ckpt", "-q"], timeout=1800)
                if r.returncode != 0:
                    r = run_quiet([self.k, "datasets", "create", "-p", stage, "-q"], timeout=1800)
                if r.returncode == 0:
                    last_mtime = m0
                    log(f"checkpoint of {name} seed {seed} saved to Kaggle")
                else:
                    log(f"checkpoint upload failed: {(r.stderr or r.stdout)[-200:].strip()}")
            except OSError as e:
                log(f"checkpoint upload skipped: {e}")

    # ------------------------------------------------------------------ main loop
    def run_seed(self, job: dict, seed: int) -> int:
        cmd = seed_command(job, seed, self.a.work_root, self.pip_done)
        self.restore_ckpt(job["name"], seed)
        log(f"START {job['name']} seed {seed}: {' '.join(cmd[1:])}")
        tail: list[str] = []
        stop = threading.Event()
        uploader = threading.Thread(target=self.ckpt_uploader, args=(job["name"], seed, stop), daemon=True)
        uploader.start()
        with subprocess.Popen(cmd, cwd=self.a.code_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            for line in p.stdout:
                sys.stdout.write(line)
                tail.append(line)
                del tail[:-400]
            rc = p.wait()
        stop.set()
        self.pip_done = True
        if rc == 0 and not self.ensure_published(job["name"], seed):
            rc = 3  # trained but not on Kaggle: keep the local copy, report it, do not count the seed as done
        key = (job["name"], seed)
        (self.trained if rc == 0 else self.failed).add(key)
        log(f"{'DONE' if rc == 0 else 'FAILED'} {job['name']} seed {seed} (exit code {rc})")
        self.report("seed_done" if rc == 0 else "seed_failed", job["name"], seed, rc, "".join(tail))
        return rc

    def ensure_published(self, name: str, seed: int, wait_minutes: float = 45) -> bool:
        """Kaggle can accept an upload and never materialise the dataset: poll, re-upload once, give up loudly."""
        runs = os.path.join(self.a.work_root, f"work-{name}", "runs")
        slug = f"{self.a.user}/rsna-knee-runs-{name}"
        for attempt in (1, 2):
            t0 = time.time()
            while time.time() - t0 < wait_minutes * 60:
                r = run_quiet([self.k, "datasets", "files", slug, "--csv", "--page-size", "200"], timeout=120)
                if r.returncode == 0 and seed in parse_done_seeds(r.stdout):
                    log(f"{slug} has seed {seed}")
                    return True
                time.sleep(60)
            if attempt == 1:
                log(f"{slug} still lacks seed {seed} after {wait_minutes:.0f} min: uploading again")
                r = run_quiet([self.k, "datasets", "version", "-p", runs, "-m", f"re-upload seed {seed}", "--dir-mode", "zip", "-q"], timeout=3600)
                if r.returncode != 0:
                    r = run_quiet([self.k, "datasets", "create", "-p", runs, "--dir-mode", "zip", "-q"], timeout=3600)
                log(f"re-upload exit code {r.returncode}: {(r.stderr or r.stdout)[-300:].strip()}")
        log(f"PUBLISH FAILED: {slug} never showed seed {seed}; the weights stay in {runs}")
        return False

    def gpu(self) -> str:
        r = run_quiet(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=60)
        return r.stdout.strip() if r.returncode == 0 else ""

    def loop(self) -> None:
        t0, last_work, fast_fail = time.time(), time.time(), 0
        gpu = self.gpu()
        log(f"worker {self.letter} up on {gpu or 'NO GPU'}; jobs from {self.a.user}/{self.a.jobs_dataset}; idle limit {self.a.idle_minutes:.0f} min")
        if not gpu and not self.a.allow_cpu:  # a CPU runtime would only fail every job (and get OOM-killed on 2 cores)
            self.report("no_gpu", tail="this runtime has no GPU: Runtime -> Change runtime type -> A100 (or L4), then rerun the cell")
            log("NO GPU in this runtime: change the runtime type to A100 (or L4) and run the cell again")
            return
        self.report("started", tail=gpu)
        while (time.time() - t0) / 3600 < self.a.max_hours:
            queue = self.fetch_queue()
            if self.letter in [str(x).upper() for x in queue.get("stop_workers", [])]:
                log(f"the queue asks worker {self.letter} to stop")
                break
            jobs = queue.get("jobs", [])
            pick = pick_job(jobs, self.letter, self.done_seeds, self.trained | self.failed)
            if pick is None:
                idle = (time.time() - last_work) / 60
                if idle >= self.a.idle_minutes or self.a.once:
                    log(f"no jobs for worker {self.letter} for {idle:.0f} min: stopping")
                    break
                log(f"no jobs for worker {self.letter} (idle {idle:.0f} min), next check in {self.a.poll_minutes:.0f} min")
                time.sleep(self.a.poll_minutes * 60)
                continue
            job, missing = pick
            self.refresh_code()
            self.restore_runs(job["name"], self.done_seeds(job["name"]))
            t_seed = time.time()
            rc = self.run_seed(job, missing[0])  # one seed, then re-read the queue: new orders apply between seeds
            last_work = time.time()
            fast_fail = fast_fail + 1 if (rc != 0 and time.time() - t_seed < 600) else 0
            if fast_fail >= 2:
                log("two seeds in a row failed within minutes: the runtime is broken, stopping (see the log above)")
                break
        self.report("stopped")
        if not self.a.no_unassign:
            log("releasing the runtime")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:  # noqa: BLE001  (not on Colab)
                log(f"unassign skipped: {e}")


def worker_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, help="letter of this session; it runs only jobs with the same letter")
    ap.add_argument("--user", default="kirillkholkin")
    ap.add_argument("--jobs-dataset", default="rsna-knee-jobs")
    ap.add_argument("--work-root", default="/content/work")
    ap.add_argument("--code-dir", default="/content/code")
    ap.add_argument("--idle-minutes", type=float, default=15)
    ap.add_argument("--poll-minutes", type=float, default=5)
    ap.add_argument("--max-hours", type=float, default=23)
    ap.add_argument("--no-unassign", action="store_true")
    ap.add_argument("--once", action="store_true", help="stop as soon as there is nothing to do (tests)")
    ap.add_argument("--allow-cpu", action="store_true", help="tests only")
    ap.add_argument("--ckpt-minutes", type=float, default=20, help="upload the rolling checkpoint this often")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    Worker(worker_args(argv)).loop()


if __name__ == "__main__":
    main()
