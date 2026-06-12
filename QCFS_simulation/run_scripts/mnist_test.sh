#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python main_test.py --model cnn2 -b 128 -j 4 -L 8 -T 0
