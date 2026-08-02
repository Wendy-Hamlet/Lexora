#!/usr/bin/env bash
# The live-pitch demo, for a bash shell (Git Bash). See demo.ps1 for PowerShell.
#
# Why this file exists: every command handed to the operator during development was
# written in bash, and Windows' default terminal is PowerShell, where `VAR=value cmd`
# is not a variable assignment but an unknown-command error. Rather than translate
# under time pressure on the day, both shells get a checked-in script that has been
# run on this machine.
#
# Usage:   ./demo.sh            one law, the pitch demo, ~12s
#          ./demo.sh sg         a whole economy from the recording, ~67s
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/Scripts/python.exe
export LEXORA_HTTP_CACHE=replay      # offline: no socket is opened, nothing is billed
export LEXORA_MY_ENUMERATE=1
export LEXORA_OCR=1

if [ "${1:-demo}" = "demo" ]; then
    # The provision the legal group chose: Money Services Business Act 2011 s.28(1),
    # P7-I3, "for a period of not less than seven years".
    export LEXORA_ONLY_LAW="MONEY SERVICES BUSINESS ACT 2011"
    exec "$PY" scripts/run_submission.py -j my --verify-clauses \
        --rationale-llm --metadata-llm \
        --no-amendment-discovery --no-child-regulations --no-regulator-instruments \
        --out outputs/demo.csv
fi

# A whole economy. Unset the filter explicitly: leaving it set would quietly turn a
# full run into a one-law run, and the row count is the only symptom.
unset LEXORA_ONLY_LAW
exec "$PY" scripts/run_submission.py -j "$1" --verify-clauses --out "outputs/live_$1.csv"
