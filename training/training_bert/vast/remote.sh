#!/usr/bin/env bash
# Local helper for the Vast box (run from the repo root in Git Bash):
#   training/training_bert/vast/remote.sh push                 copy code + data to the box (no runs, no weights)
#   training/training_bert/vast/remote.sh setup                install deps, cache the encoder, check the GPU
#   training/training_bert/vast/remote.sh run <queue.sh>       start a queue script (path in the repo) detached on the box
#   training/training_bert/vast/remote.sh pull <experiment>    copy training/training_bert/<experiment>/runs back (weights excluded)
#   training/training_bert/vast/remote.sh ssh [cmd]            shell / one command on the box
# The box address comes from VAST_SSH, e.g. "ssh -i ~/.ssh/vast_ed25519 -o IdentitiesOnly=yes -p PORT root@HOST"
# (vastai ssh-url <id> prints host and port). Queue scripts write into training/training_bert/<experiment>/runs.
# Experiment 1 ran its queues from training/training_bert/vast/queues (now experiment1/queues) into training/training_bert/runs.
set -euo pipefail
: "${VAST_SSH:?set VAST_SSH, e.g. export VAST_SSH='ssh -i ~/.ssh/vast_ed25519 -o IdentitiesOnly=yes -p 12345 root@1.2.3.4'}"
R=/workspace/y2
SSH="$VAST_SSH -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
case "${1:-}" in
  push)
    tar czf - --exclude=__pycache__ --exclude='*/runs' --exclude='*.pt' \
      src/__init__.py src/extraction yad2_search_taxonomy.json training/training_frozen_fusion/shared \
      training/training_frozen_fusion/experiment_1/artifacts/labels/val_split.json training/training_bert \
      | $SSH "mkdir -p $R && tar xzf - -C $R" ;;
  setup)
    $SSH "cd $R && bash training/training_bert/vast/setup.sh" ;;
  run)
    q=${2:?queue script path}; log=$(dirname "$q")/../runs/queue_$(basename "$q" .sh).log
    $SSH "cd $R && mkdir -p \$(dirname $log) && (setsid env LR=${LR:-} bash $q > $log 2>&1 < /dev/null &) && echo started $q" ;;
  pull)
    e=${2:?experiment folder, e.g. experiment2}; mkdir -p "training/training_bert/$e"
    $SSH "cd $R/training/training_bert/$e && tar czf - --exclude='*.pt' runs" | tar --force-local -xzf - -C "training/training_bert/$e" ;;
  ssh)
    shift; $SSH "$@" ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
