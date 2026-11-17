# E3: seed variance at the selected LR (LR=... from the E2 sweep); seed 42 is the E2 run itself
set -uo pipefail
: "${LR:?set LR}"
for seed in 13 1337; do
  out=training/training_bert/runs/e3_lr${LR}_s${seed}
  [ -f $out/metrics.json ] && grep -q '"epoch": 8' $out/metrics.json && continue
  python -m training.training_bert.train --out $out --lr $LR --seed $seed --epochs 8 > $out.log 2>&1 || echo "FAILED $out"
done
echo E3 done
