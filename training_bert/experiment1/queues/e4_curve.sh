# E4: learning curve at the selected LR. Nested family-grouped subsets of train (subset seed s gives one nested
# chain; two chains x two training seeds), plus silver-only. The full-train point is E2/E3.
set -uo pipefail
: "${LR:?set LR}"
for pair in "0 42" "1 13"; do
  set -- $pair; sub=$1; seed=$2
  for n in 2000 4000 8000 12000; do
    out=training_bert/runs/e4_n${n}_sub${sub}_s${seed}
    [ -f $out/metrics.json ] && grep -q '"epoch": 8' $out/metrics.json && continue
    python -m training_bert.train --out $out --lr $LR --seed $seed --epochs 8 --train-size $n --subset-seed $sub \
      > $out.log 2>&1 || echo "FAILED $out"
  done
  out=training_bert/runs/e4_silver_s${seed}
  [ -f $out/metrics.json ] && grep -q '"epoch": 8' $out/metrics.json || \
    python -m training_bert.train --out $out --lr $LR --seed $seed --epochs 8 --quality silver > $out.log 2>&1 \
    || echo "FAILED $out"
done
echo E4 done
