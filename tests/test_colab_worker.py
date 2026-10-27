import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab"))
import worker  # noqa: E402


def test_parse_done_seeds():
    csv_out = "name,size,creationDate\nfold0/best.pt,1,x\nfold0/history.jsonl,1,x\nfold2/best.pt,1,x\nfold1/config.yaml,1,x\n"
    assert worker.parse_done_seeds(csv_out) == {0, 2}
    assert worker.parse_done_seeds("") == set()


def test_pick_job_order_worker_letter_skip_and_disabled():
    jobs = [{"name": "a", "worker": "B", "seeds": [0]},
            {"name": "b", "worker": "A", "seeds": [0, 1], "disabled": True},
            {"name": "c", "worker": "a", "seeds": [0, 1, 2]},
            {"name": "d", "worker": "A", "seeds": [0]}]
    done = {"c": {0}}
    pick = worker.pick_job(jobs, "A", lambda n: done.get(n, set()), skip={("c", 1)})
    assert pick[0]["name"] == "c" and pick[1] == [2]
    assert worker.pick_job(jobs, "A", lambda n: {0, 1, 2}, skip=set()) is None


def test_seed_command_sets_its_own_flags_and_rejects_duplicates():
    job = {"name": "v1", "args": ["--config", "c.yaml", "--override", "optim.epochs=20"]}
    cmd = worker.seed_command(job, 2, "/content/work", skip_pip=True)
    assert cmd[1:] == ["colab/colab_train.py", "--name", "v1", "--folds", "2", "--drive-root", "/content/work",
                       "--config", "c.yaml", "--override", "optim.epochs=20", "--upload", "--skip-pip"]
    with pytest.raises(ValueError):
        worker.seed_command({"name": "v1", "args": ["--upload"]}, 0, "/w", False)


def test_loop_trains_missing_seeds_then_releases(monkeypatch, tmp_path):
    jobs = [{"name": "j", "worker": "A", "seeds": [0, 1]}]
    w = worker.Worker(worker.worker_args(["--worker", "a", "--once", "--no-unassign", "--allow-cpu", "--work-root", str(tmp_path)]))
    calls = []
    monkeypatch.setattr(w, "fetch_queue", lambda: {"jobs": jobs})
    monkeypatch.setattr(w, "refresh_code", lambda: None)
    monkeypatch.setattr(w, "restore_runs", lambda name, done: None)
    monkeypatch.setattr(w, "report", lambda *a, **k: None)
    monkeypatch.setattr(worker, "run_quiet", lambda cmd, timeout=600: worker.subprocess.CompletedProcess(cmd, 1, "", ""))

    def fake_run_seed(job, seed):
        calls.append(seed)
        w.trained.add((job["name"], seed))
        return 0
    monkeypatch.setattr(w, "run_seed", fake_run_seed)
    monkeypatch.setattr(w, "ensure_published", lambda name, seed: True)
    w.loop()
    assert calls == [0, 1]


def test_worker_refuses_cpu_runtime(monkeypatch, tmp_path):
    w = worker.Worker(worker.worker_args(["--worker", "c", "--once", "--no-unassign", "--work-root", str(tmp_path)]))
    states = []
    monkeypatch.setattr(w, "gpu", lambda: "")
    monkeypatch.setattr(w, "report", lambda state, *a, **k: states.append(state))
    monkeypatch.setattr(w, "fetch_queue", lambda: (_ for _ in ()).throw(AssertionError("must not fetch jobs without a GPU")))
    w.loop()
    assert states == ["no_gpu"]


def test_ckpt_slug_and_restore(monkeypatch, tmp_path):
    assert worker.ckpt_slug("v14-nano-long-smart", 2) == "rsna-knee-ckpt-v14-nano-long-smart-s2"
    assert len(worker.ckpt_slug("x" * 80, 0)) == 50
    w = worker.Worker(worker.worker_args(["--worker", "a", "--no-unassign", "--work-root", str(tmp_path)]))

    def fake_download(cmd, timeout=600):
        dest = cmd[cmd.index("-p") + 1]
        os.makedirs(dest, exist_ok=True)
        for f in ("last.pt", "history.jsonl"):
            open(os.path.join(dest, f), "w").write("x")
        return worker.subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(worker, "run_quiet", fake_download)
    monkeypatch.setattr(worker.shutil, "rmtree", lambda *a, **k: None)
    w.restore_ckpt("j", 1)
    fold = w.fold_dir("j", 1)
    assert os.path.exists(os.path.join(fold, "last.pt")) and os.path.exists(os.path.join(fold, "history.jsonl"))


def test_stop_flag_releases_worker(monkeypatch, tmp_path):
    w = worker.Worker(worker.worker_args(["--worker", "a", "--no-unassign", "--allow-cpu", "--work-root", str(tmp_path)]))
    monkeypatch.setattr(w, "report", lambda *a, **k: None)
    monkeypatch.setattr(w, "fetch_queue", lambda: {"stop_workers": ["A"], "jobs": [{"name": "j", "worker": "A", "seeds": [0]}]})
    monkeypatch.setattr(w, "run_seed", lambda *a: (_ for _ in ()).throw(AssertionError("must not train after a stop")))
    w.loop()
