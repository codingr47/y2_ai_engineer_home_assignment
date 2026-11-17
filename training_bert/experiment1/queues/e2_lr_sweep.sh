# E2: encoder learning rate, seed 42, full train, 8 epochs (3e-5 is the E1 baseline)
set -uo pipefail
for lr in 3e-5 1e-5 2e-5 5e-5; do
  out=training_bert/runs/e2_lr${lr}_s42
  [ -f $out/metrics.json ] && grep -q '"epoch": 8' $out/metrics.json && continue
  python -m training_bert.train --out $out --lr $lr --seed 42 --epochs 8 > $out.log 2>&1 || echo "FAILED $out"
done
echo E2 done
