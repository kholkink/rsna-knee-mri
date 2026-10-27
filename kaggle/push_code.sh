#!/usr/bin/env bash
# Package the repo (no venv / runs / cache) as the Kaggle dataset "<user>/rsna-knee-code" and create or update it.
# Usage: bash kaggle/push_code.sh            (needs ~/.kaggle/kaggle.json)
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=$(python kaggle/launch.py whoami)
STAGE=$(mktemp -d)
rsync -a --exclude '/.venv' --exclude '/.env' --exclude '/runs' --exclude '/cache' --exclude '/features' --exclude '/data' \
      --exclude '/labels' --exclude '/kaggle_out' --exclude '/build' --exclude '/.git' --exclude '__pycache__' --exclude '.pytest_cache' \
      --exclude '.ruff_cache' --exclude '*.egg-info' \
      ./ "$STAGE/"
for must in rsna_knee/data/dataset.py rsna_knee/train.py configs/labels/label_table_blend_v2.parquet configs/folds_v1.parquet colab/colab_train.py kaggle/launch.py kaggle/kernels/train_fold.py; do
  [ -f "$STAGE/$must" ] || { echo "MISSING in staged code: $must"; exit 1; }
done
grep -rIl "sk-[0-9a-f]\{20,\}" "$STAGE" && { echo "secret found in staged code"; exit 1; } || true
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "rsna-knee-code", "id": "${USER_NAME}/rsna-knee-code", "licenses": [{"name": "MIT"}]}
JSON
if kaggle datasets status "${USER_NAME}/rsna-knee-code" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "update $(date +%F_%H%M)" --dir-mode zip
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
rm -rf "$STAGE"
echo "dataset: https://www.kaggle.com/datasets/${USER_NAME}/rsna-knee-code"
