# Evaluation interpretation (auto-generated)

* Test accuracy **0.997**, macro precision 0.997, macro recall 0.997, macro F1 0.997 on 360 unseen samples.
* Validation accuracy 1.000 vs test 0.997 (gap +0.003): a small gap means no strong over-fitting to the validation set.
* Weakest class: **peace** (F1 0.989).
* Most frequent confusion: **peace -> pointing** (1 samples).
* With the app threshold 0.80, 100.0% of test frames are accepted and 99.7% of those are correct - this is why 0.80 + a 2 s hold is used to suppress wrong momentary predictions.
* Inference latency: 8.61 ms per frame (real-time capable).

Caveat: results come from data recorded by few people in few sessions; accuracy for new users, lighting or camera angles can be lower. Collect more varied data to improve generalisation.