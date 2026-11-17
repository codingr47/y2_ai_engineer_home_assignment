# E6: the selected config (LR 5e-5, 8 epochs, fixed on val_eval) retrained with saved weights, 3 seeds, then the
# val_eval / val_calib scoring (test is held back for the final model: final_test --with-test)
set -uo pipefail
runs=()
for seed in 42 13 1337; do
  out=training_bert/runs/e6_lr5e-5_s${seed}
  runs+=($out)
  [ -f $out/model.pt ] && grep -q '"epoch": 8' $out/metrics.json && continue
  python -m training_bert.train --out $out --lr 5e-5 --seed $seed --epochs 8 --save > $out.log 2>&1 || echo "FAILED $out"
done
python -m training_bert.final_test "${runs[@]}" > training_bert/runs/e6_final_test.log 2>&1 || echo "FAILED final_test"
echo E6 done
