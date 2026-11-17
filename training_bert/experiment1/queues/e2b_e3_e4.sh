# E2 boundary check (8e-5: 5e-5 won at the grid edge), then E3 seeds and E4 learning curve at LR
set -uo pipefail
out=training_bert/runs/e2_lr8e-5_s42
[ -f $out/metrics.json ] && grep -q '"epoch": 8' $out/metrics.json || \
  python -m training_bert.train --out $out --lr 8e-5 --seed 42 --epochs 8 > $out.log 2>&1 || echo "FAILED $out"
bash training_bert/vast/queues/e3_seeds.sh
bash training_bert/vast/queues/e4_curve.sh
echo ALL done
