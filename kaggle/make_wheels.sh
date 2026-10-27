#!/usr/bin/env bash
# Download wheels for the offline notebook (run on a machine with internet, upload dir as a Kaggle dataset).
set -euo pipefail
OUT=${1:-wheels}
mkdir -p "$OUT"
pip download --dest "$OUT" --only-binary=:all: --python-version 3.11 --platform manylinux2014_x86_64 \
  pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg timm
echo "wheels -> $OUT"
