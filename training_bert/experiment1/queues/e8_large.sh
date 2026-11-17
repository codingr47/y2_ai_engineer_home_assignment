# E8: DictaBERT-large (24 layers, 1024 hidden, same tokenizer). Large encoders usually want a lower LR than base;
# 3e-5 first, then 2e-5. 8 epochs, saved weights, scored with the improved normalizer.
set -uo pipefail
for lr in 3e-5 2e-5; do
  out=training_bert/runs/e8_large_lr${lr}_s42
  [ -f $out/metrics.json ] && grep -q '"epoch": 8' $out/metrics.json && continue
  python -m training_bert.train --out $out --encoder dicta-il/dictabert-large --lr $lr --seed 42 --epochs 8 --save \
    > $out.log 2>&1 || echo "FAILED $out"
done
echo E8 done
