#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python main_train.py --model cnn2 --epochs 100 -lr 0.01 -b 128 -j 4 -L 8 -T 0
