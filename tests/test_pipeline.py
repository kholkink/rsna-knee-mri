"""End-to-end on synthetic DICOM: scan -> select -> cache -> labels -> folds -> train 1 epoch -> predict -> submission."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.config import DataConfig, ModelConfig, OptimConfig, TrainConfig
from rsna_knee.constants import LABELS
from rsna_knee.cv import make_folds, study_groups
from rsna_knee.data.dataset import StudyDataset, collate_studies
from rsna_knee.dicom import SeriesMeta, scan_series_header
from rsna_knee.infer import InferenceConfig, predict, write_submission
from rsna_knee.models import SoftTargetBCE, StudyModel
from rsna_knee.preprocess import CacheJob, build_cache, load_series_npz
from rsna_knee.reports.labels import LabelMapConfig, build_label_table
from rsna_knee.reports.schema import Finding, empty_extraction
from rsna_knee.series_select import SelectionPolicy, select_series
from rsna_knee.train import build_records, train_fold

LABEL_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "Patellofemoral OA",
              "Effusion", "Synovitis", "Bakers Cyst", "Contusion", "Fracture"]


@pytest.fixture(scope="session")
def pipeline(synthetic_root, tmp_path_factory):
    root, studies = synthetic_root
    work = tmp_path_factory.mktemp("work")
    # 1. metadata scan + selection
    metas = []
    for uid, series in studies.items():
        for suid in series:
            metas.append(scan_series_header(os.path.join(root, uid, suid), uid, suid))
    meta_df = pd.DataFrame([m.to_dict() for m in metas])
    assert meta_df["plane"].isin(["sagittal", "coronal", "axial"]).all()
    jobs = []
    for uid, g in meta_df.groupby("study_uid"):
        chosen = select_series([SeriesMeta.from_dict(r) for r in g.to_dict("records")], SelectionPolicy())
        assert [m.plane for m in chosen] == ["sagittal", "coronal", "axial"]  # localizer dropped
        for m in chosen:
            jobs.append(CacheJob(m, str(work / "cache" / uid / f"{m.series_uid}.npz"), size=48, max_slices=6))
    # 2. cache
    idx = build_cache(jobs, workers=1, progress=False)
    assert (idx["status"] == "ok").all()
    vol, pos = load_series_npz(idx["path"][0], [0, 2])
    assert vol.shape == (2, 48, 48) and len(pos) == 6
    idx.to_parquet(work / "cache_index.parquet", index=False)
    # 3. labels: fake extractions + one gold study
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"StudyInstanceUID": list(studies), "Report": ["r"] * len(studies)})
    for c in LABEL_COLS:
        train[c] = np.nan
    train.loc[0, LABEL_COLS] = rng.integers(0, 2, len(LABEL_COLS)).astype(float)
    ext = {}
    for uid in studies:
        e = empty_extraction("en")
        for k in LABELS:
            st = rng.choice(["present", "absent", "equivocal", "not_mentioned"])
            setattr(e, k, Finding(status=st, severity="unknown", detail="", evidence=""))
        ext[uid] = e
    table = build_label_table(train, ext, LabelMapConfig())
    table.to_parquet(work / "label_table.parquet", index=False)
    folds = make_folds(table, study_groups(meta_df), n_folds=3, seed=0)
    assert folds["fold"].nunique() == 3
    folds.to_parquet(work / "folds.parquet", index=False)
    return root, work, train


def test_dataset_and_model_forward(pipeline):
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=5),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1))
    recs = build_records(cfg, "all")
    assert len(recs) == 6 and all(len(r.series) == 3 for r in recs)
    ds = StudyDataset(recs, 48, 4, train=True, max_series=3)
    batch = collate_studies([ds[0], ds[1]])
    assert batch["images"].shape == (6, 4, 3, 48, 48) and batch["images"].dtype == torch.uint8
    assert batch["series_batch_idx"].tolist() == [0, 0, 0, 1, 1, 1]
    model = StudyModel("resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1)
    out = model(batch["images"], batch["slice_pos"], batch["plane_ids"], batch["seq_ids"], batch["series_batch_idx"], 2, return_attn=True)
    assert out["logits"].shape == (2, 12) and out["attn"].shape == (2, 12, 1 + 3 * 4)
    loss = SoftTargetBCE()(out["logits"], batch["targets"], batch["weights"])
    loss.backward()
    assert torch.isfinite(loss)
    # a study with zero series must still produce finite logits (study token only)
    empty = {"images": torch.zeros((0, 4, 3, 48, 48), dtype=torch.uint8), "slice_pos": torch.zeros((0, 4)),
             "plane_ids": torch.zeros(0, dtype=torch.long), "seq_ids": torch.zeros(0, dtype=torch.long),
             "series_batch_idx": torch.zeros(0, dtype=torch.long)}
    assert torch.isfinite(model(batch_size=1, **empty)["logits"]).all()
    # precomputed-feature mode
    fm = StudyModel(feature_dim=16, pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1)
    assert fm(torch.randn(3, 4, 16), torch.rand(3, 4), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
              torch.tensor([0, 0, 1]), 2)["logits"].shape == (2, 12)


def test_train_and_predict_end_to_end(pipeline, tmp_path):
    root, work, train = pipeline
    cfg = TrainConfig(
        data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                        folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=5, num_workers=0),
        model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1),
        optim=OptimConfig(epochs=2, batch_size=1, accum=1, amp="off", ema_decay=0.5, lr_backbone=1e-3, lr_head=1e-3),
        fold=0, output_dir=str(tmp_path / "run"), resume=True)
    res = train_fold(cfg)
    run_dir = res["out_dir"]
    assert os.path.exists(os.path.join(run_dir, "best.pt")) and os.path.exists(os.path.join(run_dir, "oof.parquet"))
    assert len(res["history"]) == 2
    # resume: nothing left to train, must not crash
    res2 = train_fold(cfg)
    assert res2["best_auc"] == res["best_auc"]

    # fake competition root: test.csv + sample_submission + test images (reuse synthetic studies)
    comp = tmp_path / "comp"
    comp.mkdir()
    os.symlink(root, comp / "test_images")
    uids = list(train["StudyInstanceUID"])[:3]
    pd.DataFrame({"StudyInstanceUID": uids}).to_csv(comp / "test.csv", index=False)
    sub = pd.DataFrame({"StudyInstanceUID": uids, **{c: 0.5 for c in LABEL_COLS}})
    sub.to_csv(comp / "sample_submission.csv", index=False)
    icfg = InferenceConfig(comp_root=str(comp), run_dirs=[run_dir], tta_hflip=True, batch_size=2, num_workers=0)
    pred = predict(icfg)
    assert len(pred) == 3 and ((pred[LABELS] >= 0) & (pred[LABELS] <= 1)).all().all()
    out = write_submission(pred, str(comp / "sample_submission.csv"), str(tmp_path / "submission.csv"))
    assert list(out.columns) == ["StudyInstanceUID"] + LABEL_COLS
    assert not np.allclose(out[LABEL_COLS].values, 0.5)


def test_spatial_branch_is_neutral_at_init_and_trains():
    torch.manual_seed(0)
    kw = dict(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1, dropout=0.0, drop_path=0.0)
    base, sp = StudyModel(**kw), StudyModel(spatial_head=True, spatial_dim=24, **kw)
    sp.load_state_dict(base.state_dict(), strict=False)
    base.eval(); sp.eval()
    imgs = torch.randint(0, 255, (3, 4, 3, 64, 64), dtype=torch.uint8)
    args = (imgs, torch.rand(3, 4), torch.tensor([0, 1, 2]), torch.tensor([1, 6, 3]), torch.tensor([0, 0, 1]), 2)
    o_base, o_sp = base(*args), sp(*args, return_attn=True)
    assert torch.allclose(o_base["logits"], o_sp["logits"], atol=1e-5)          # zero-initialised output -> same function
    assert o_sp["cell_att"].shape == (3, 4, 12, 4) and o_sp["slice_att"].shape == (2, 12, 8)   # 64px / 32 = 2x2 cells
    att = o_sp["slice_att"]
    assert torch.allclose(att.sum(-1), torch.ones(2, 12), atol=1e-5) and float(att[1, :, 4:].abs().max()) == 0.0  # padded series masked
    sp.train()
    loss = SoftTargetBCE()(sp(*args)["logits"], torch.rand(2, 12), torch.ones(2, 12))
    loss.backward()
    assert sp.spatial.out_w.grad is not None and float(sp.spatial.out_w.grad.abs().sum()) > 0
    # ViT backbones expose tokens instead of a map
    vit = StudyModel(backbone="vit_tiny_patch16_224", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1,
                     spatial_head=True, spatial_dim=24)
    v = vit(torch.randint(0, 255, (1, 2, 3, 224, 224), dtype=torch.uint8), torch.rand(1, 2), torch.tensor([0]), torch.tensor([1]),
            torch.tensor([0]), 1, return_attn=True)
    assert v["logits"].shape == (1, 12) and v["cell_att"].shape == (1, 2, 12, 196)


def test_window_budget_padding_and_topk_average(pipeline, tmp_path):
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=5),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1))
    recs = build_records(cfg, "all")
    recs[0].series[0].n_slices = 12  # pretend one series is twice as long -> unequal window budgets inside the study
    ds = StudyDataset(recs, 48, 4, train=False, max_series=3, windows_per_study=9)
    budgets = ds._window_budget(recs[0].series)
    assert sum(budgets) == 9 and min(budgets) >= 2 and len(set(budgets)) > 1
    it = ds[0]                                                        # flat bag: one pseudo-series of exactly K windows
    assert it["images"].shape == (1, 9, 3, 48, 48) and it["n_series"] == 1 and int(it["slice_valid"].sum()) == 9
    assert it["plane_ids"].shape == (1, 9) and sorted(set(it["plane_ids"][0].tolist())) == [0, 1, 2]
    assert [int((it["plane_ids"][0] == p).sum()) for p in (0, 1, 2)] == budgets
    batch = collate_studies([ds[0], ds[1]])
    assert batch["images"].shape == (2, 9, 3, 48, 48) and batch["series_batch_idx"].tolist() == [0, 1]   # constant batch shape
    batch["slice_valid"][0, -2:] = False                                # as if a series had failed to load
    model = StudyModel("resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=1, study_layers=1, spatial_head=True, spatial_dim=16)
    out = model(batch["images"], batch["slice_pos"], batch["plane_ids"], batch["seq_ids"], batch["series_batch_idx"], 2,
                return_attn=True, slice_valid=batch["slice_valid"])
    assert out["logits"].shape == (2, 12)
    att = out["attn"][:, :, 1:]                                         # attention over masked windows must be zero
    assert float(att[0][:, -2:].abs().max()) == 0.0 and float(att[1][:, -2:].abs().max()) > 0.0
    assert float(out["slice_att"][0][:, -2:].abs().max()) == 0.0
    # train-on-all + top-k averaging + select_by=last
    from rsna_knee.train import average_state_dicts
    cfg2 = TrainConfig(data=cfg.data, model=cfg.model, optim=OptimConfig(epochs=2, batch_size=1, accum=1, amp="off", ema_decay=0.5, topk_avg=2),
                       train_on_all=True, select_by="last", output_dir=str(tmp_path / "all"), resume=False)
    cfg2.data.num_workers = 0; cfg2.data.windows_per_study = 6
    res = train_fold(cfg2)
    assert os.path.exists(os.path.join(res["out_dir"], "avg_top2.pt")) and os.path.exists(os.path.join(res["out_dir"], "best.pt"))
    tr_recs, va_recs = build_records(cfg2, "train"), build_records(cfg2, "val")
    assert all(not r.is_gold for r in tr_recs) and all(r.is_gold for r in va_recs) and len(va_recs) == 1
    sd = average_state_dicts([model.state_dict(), model.state_dict()])
    assert torch.allclose(sd["head_b"], model.state_dict()["head_b"])


def test_simple_head_without_transformers():
    m = StudyModel("resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0)
    assert m.series_enc is None and m.study_enc is None
    x = torch.randint(0, 255, (2, 5, 3, 48, 48), dtype=torch.uint8)
    valid = torch.ones(2, 5, dtype=torch.bool); valid[0, -2:] = False
    out = m(x, torch.rand(2, 5), torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.long), torch.tensor([0, 1]), 2,
            return_attn=True, slice_valid=valid)
    assert out["logits"].shape == (2, 12) and float(out["attn"][0][:, 1:][:, -2:].abs().max()) == 0.0


def test_gold_holdout_protocol(pipeline, tmp_path):
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                      optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), gold_holdout=True, select_by="confident", fold=1,
                      output_dir=str(tmp_path / "gh"), resume=False)
    import pandas as pd
    folds = pd.read_parquet(work / "folds.parquet").set_index("study_uid")["fold"]
    tr, va = build_records(cfg, "train", 1), build_records(cfg, "val", 1)
    assert not any(r.is_gold for r in tr) and all(folds[r.study_uid] != 1 for r in tr)
    assert sum(r.is_gold for r in va) == 1 and all(r.is_gold or folds[r.study_uid] == 1 for r in va)
    assert len(tr) + len(va) == 6 and not {r.study_uid for r in tr} & {r.study_uid for r in va}
    res = train_fold(cfg)
    h = res["history"][-1]
    assert h["n_val"] == len(va) - 1 and h["gold_n"] == 1 and "confident_per_label" in h


def test_mix_labels_only_touches_targets():
    import numpy as np

    from rsna_knee.constants import LABELS
    from rsna_knee.infer import mix_labels

    rng = np.random.default_rng(0)
    final = rng.random((20, len(LABELS)))
    out = mix_labels(final, {"synovitis": {"effusion": 0.5}})
    js, je = LABELS.index("synovitis"), LABELS.index("effusion")
    untouched = [j for j in range(len(LABELS)) if j != js]
    assert np.array_equal(out[:, untouched], final[:, untouched])
    top_eff = int(np.argmax(final[:, je]))
    assert out[top_eff, js] >= 0.5  # the study with most effusion is pulled up for synovitis
    assert mix_labels(final, {}) is final


def test_gold_training_cv_splits(pipeline, tmp_path):
    import pandas as pd

    _, work, _ = pipeline
    table = pd.read_parquet(work / "label_table.parquet")
    table.loc[table.index[:3], "is_gold"] = True  # three expert studies -> three parts of one study each
    table.to_parquet(tmp_path / "labels_gold3.parquet", index=False)
    gold = set(table.loc[table["is_gold"], "study_uid"].astype(str))

    def cfg_for(fold, cv):
        return TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(tmp_path / "labels_gold3.parquet"),
                                           folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                           model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                           optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), train_on_all=True, gold_train=True, gold_cv=cv,
                           gold_parts=3, select_by="last", fold=fold, output_dir=str(tmp_path / f"gcv{fold}{cv}"), resume=False)

    held = []
    for fold in range(3):
        tr, va = build_records(cfg_for(fold, True), "train", fold), build_records(cfg_for(fold, True), "val", fold)
        assert len(va) == 1 and va[0].is_gold and va[0].study_uid not in {r.study_uid for r in tr}
        assert sum(r.is_gold for r in tr) == 2 and len(tr) == 5
        held.append(va[0].study_uid)
    assert set(held) == gold  # every expert study is held out exactly once
    tr, va = build_records(cfg_for(0, False), "train"), build_records(cfg_for(0, False), "val")
    assert {r.study_uid for r in va} == gold and sum(r.is_gold for r in tr) == 3  # final models: all gold trains
    res = train_fold(cfg_for(1, True))
    assert res["history"][-1]["gold_n"] == 1


def test_eval_variants_script(pipeline, tmp_path, monkeypatch):
    import runpy
    import sys

    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                      optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), gold_holdout=True, select_by="confident", fold=1,
                      output_dir=str(tmp_path / "ev"), resume=False)
    res = train_fold(cfg)
    out = tmp_path / "variants"
    monkeypatch.setattr(sys, "argv", ["eval_variants.py", "--run", res["out_dir"], "--windows", "4", "6", "--offsets", "0", "0.5",
                                      "--workers", "0", "--out", str(out)])
    runpy.run_path("scripts/eval_variants.py", run_name="__main__")
    import pandas as pd
    rep = pd.read_csv(f"{out}.csv")
    assert set(rep["pool"]) == {"mean", "max for focal"} and len(rep) == 2 * 3


def test_queue_pairwise_loss_ranks_confident_cells_only():
    import torch

    from rsna_knee.models import QueuePairwiseLoss

    loss_fn = QueuePairwiseLoss(n_labels=2, size=8)
    z = torch.zeros(2, 2, requires_grad=True)
    t = torch.tensor([[1.0, 0.5], [0.0, float("nan")]])
    assert float(loss_fn(z, t)) == 0.0  # empty queue: no pairs yet, and the batch is enqueued
    z2 = torch.zeros(2, 2, requires_grad=True)
    loss = loss_fn(z2, t)
    loss.backward()
    assert float(loss) > 0
    assert z2.grad[0, 0] < 0 < z2.grad[1, 0]  # the positive is pushed up, the negative down
    assert float(z2.grad[:, 1].abs().sum()) == 0.0  # mid-range and NaN targets never form pairs
    sep = torch.tensor([[6.0, 0.0], [-6.0, 0.0]])
    for _ in range(4):
        loss_fn(sep, t)  # the queue now holds well separated logits
    assert float(loss_fn(sep, t)) < 0.01


def test_training_with_pairwise_term(pipeline, tmp_path):
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                      optim=OptimConfig(epochs=2, batch_size=1, accum=1, amp="off", pair_weight=0.5, pair_last_epochs=1, pair_queue=8),
                      gold_holdout=True, select_by="confident", fold=1, output_dir=str(tmp_path / "pw"), resume=False)
    res = train_fold(cfg)
    assert len(res["history"]) == 2 and all(np.isfinite(h["train_loss"]) for h in res["history"])


def test_warm_start_loads_backbone_only(pipeline, tmp_path):
    import torch

    from rsna_knee.train import build_model, warm_start

    _, work, _ = pipeline
    def cfg_for(name, **kw):
        return TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                           folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                           model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                           optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), gold_holdout=True, select_by="confident", fold=1,
                           output_dir=str(tmp_path / name), resume=False, **kw)
    src = train_fold(cfg_for("src"))
    ck = torch.load(f"{src['out_dir']}/best.pt", map_location="cpu", weights_only=False)["model"]
    model = build_model(cfg_for("x"))
    n = warm_start(model, f"{src['out_dir']}/best.pt", backbone_only=True)
    sd = model.state_dict()
    bb = [k for k in sd if k.startswith("backbone.")]
    assert n == len(bb) and all(torch.equal(sd[k], ck[k]) for k in bb)
    assert any(not torch.equal(sd[k], ck[k]) for k in sd if not k.startswith("backbone.") and sd[k].dtype.is_floating_point)
    res = train_fold(cfg_for("warm", init_from=f"{src['out_dir']}/best.pt"))
    assert np.isfinite(res["history"][-1]["train_loss"])


def test_teacher_split_trains_on_gold(pipeline, tmp_path):
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                      optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), gold_holdout=True, gold_train=True, select_by="confident",
                      fold=1, output_dir=str(tmp_path / "teacher"), resume=False)
    tr, va = build_records(cfg, "train", 1), build_records(cfg, "val", 1)
    assert sum(r.is_gold for r in tr) == 1 and not any(r.is_gold for r in va)  # the expert study trains, the fold alone validates
    assert len(tr) + len(va) == 6 and not {r.study_uid for r in tr} & {r.study_uid for r in va}


def test_colab_pipeline_dry_run(capsys, monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--name", "vT", "--stage", "all", "--warm", "--gold", "--pair", "0.3", "--dry-run",
                                      "--drive-root", "/tmp/none"])
    runpy.run_path("colab/pipeline.py", run_name="__main__")
    out = capsys.readouterr().out
    calls = [l for l in out.splitlines() if l.startswith("$ ") and "colab_train.py" in l]
    assert len(calls) == 5 + 3
    teacher, prod = calls[0], calls[-1]
    assert "configs/v4_base.yaml" in teacher and "gold_train=true" in teacher and "rsna-knee-runs-oof-nano/fold2/best.pt" in teacher
    assert "label_table_pseudo_vT.parquet" in prod and "select_by=last" in prod and "--upload" in prod and "optim.pair_weight=0.3" in prod
    assert "rsna-knee-runs-v8-nano-pseudo-full/fold2/best.pt" in prod and "make_pseudo_labels.py" in out


def test_wide_cache_late_crop_training(pipeline, tmp_path):
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=32, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6,
                                      cache_mm=180.0, view_mm=140.0, cache_size=48, smart_crop=True, crop_jitter_mm=5.0),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0),
                      optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), gold_holdout=True, select_by="confident", fold=1,
                      output_dir=str(tmp_path / "late"), resume=False)
    recs = build_records(cfg, "all")
    from rsna_knee.data.dataset import view_kwargs
    ds = StudyDataset(recs, 32, 4, train=True, windows_per_study=6, view=view_kwargs(cfg.data))
    assert ds.late and ds[0]["images"].shape[-2:] == (32, 32)
    assert np.isfinite(train_fold(cfg)["history"][-1]["train_loss"])


def test_meanmax_pooling_head_trains_and_ignores_padding(pipeline, tmp_path):
    import torch

    from rsna_knee.models import StudyModel

    m = StudyModel(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0, pool="meanmax").eval()
    img = torch.randint(0, 255, (2, 4, 3, 48, 48), dtype=torch.uint8)
    args = (torch.rand(2, 4), torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long), torch.tensor([0, 1]), 2)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    a = m(img, *args, slice_valid=valid)["logits"]
    img2 = img.clone(); img2[0, 2:] = 0  # padded windows must not matter
    b = m(img2, *args, slice_valid=valid)["logits"]
    assert a.shape == (2, 12) and torch.allclose(a, b, atol=1e-5)
    _, work, _ = pipeline
    cfg = TrainConfig(data=DataConfig(cache_index=str(work / "cache_index.parquet"), label_table=str(work / "label_table.parquet"),
                                      folds=str(work / "folds.parquet"), img_size=48, n_slices=4, n_slices_eval=4, num_workers=0, windows_per_study=6),
                      model=ModelConfig(backbone="resnet18", pretrained=False, d_model=32, n_heads=4, series_layers=0, study_layers=0, pool="meanmax"),
                      optim=OptimConfig(epochs=1, batch_size=1, accum=1, amp="off"), gold_holdout=True, select_by="confident", fold=1,
                      output_dir=str(tmp_path / "mm"), resume=False)
    assert np.isfinite(train_fold(cfg)["history"][-1]["train_loss"])
