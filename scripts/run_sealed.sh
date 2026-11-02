#!/usr/bin/env bash
# The three sealed tests, each opened once, in the pre-registered order: Stage 1, Stage 2, crypto. Stops at the
# first failure (each script refuses to run without the prereg-v1 tag, a final spec and unchanged models).
#   setsid nohup scripts/run_sealed.sh >> logs/sealed.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
echo "[$(date +%H:%M:%S)] Stage 1";  scripts/run_stage1.sh
echo "[$(date +%H:%M:%S)] Stage 2";  scripts/run_stage2.sh
echo "[$(date +%H:%M:%S)] crypto";   scripts/run_crypto_test.sh
echo "[$(date +%H:%M:%S)] all sealed tests done"
