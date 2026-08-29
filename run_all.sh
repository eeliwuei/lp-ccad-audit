#!/usr/bin/env sh
set -eu
python3 analysis/replay_results.py
python3 -m unittest discover -s tests -v
echo "Portable analytical replay completed."
