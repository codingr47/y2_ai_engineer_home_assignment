# E7: learned categorical value head (4 value policies scored per epoch on val_eval).
# 8 epochs x seeds 42/13 compare with E3 (89.3% +- 0.4); then 12 epochs, seed 42 (5e-5 still rose at epoch 8).
# NOTE: stopped by hand after its seed-42 run (value head dropped; see experiment1/README.md)
set -uo pipefail
run() { out=$1; shift
  [ -f $out/metrics.json ] && grep -q "\"epoch\": $EP" $out/metrics.json && return
  python -m training_bert.train --out $out "$@" > $out.log 2>&1 || echo "FAILED $out"; }
# first: E6 checkpoints re-scored with the improved normalizer (number-word typos, storage sizes) = the fair
# baseline for the value head, which is scored with the same normalizer
python -m training_bert.final_test --tag=norm2 training_bert/runs/e6_lr5e-5_s13   training_bert/runs/e6_lr5e-5_s1337 > training_bert/runs/e7_rescore_e6_norm2.log 2>&1 || echo "FAILED rescore"
# seed 42 was started early, in parallel with E6 (from /workspace/y2_new); wait for it, then seed 13
while pgrep -f "e7_vh_lr5e-5_s42 " >/dev/null; do sleep 30; done
EP=8;  for seed in 13; do run training_bert/runs/e7_vh_lr5e-5_s${seed} --lr 5e-5 --seed $seed --epochs 8 --value-head --save; done
EP=12; run training_bert/runs/e7_vh_lr5e-5_ep12_s42 --lr 5e-5 --seed 42 --epochs 12 --value-head --save
echo E7 done
