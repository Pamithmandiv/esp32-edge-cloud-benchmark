# Suggested plots and evidence

1. Same 100+ independent labeled rows run on ESP32 and cloud INT8. CSV paired row_id, label, predicted_status, score, clipping, wall clock. Compute confusion matrix / balanced accuracy / abnormal recall / specificity. If labels are rule-created rather than field-verified say **proxy-label classification**, not safety detection accuracy.
2. Edge `Invoke()` duration (µs->ms), cloud server `Invoke()` duration (ms); shows compute-performance difference. Edge hardware frequency and Cloud Run CPU must be recorded.
3. Cloud HTTPS RTT including up/down in ms; use 30-100+ repeated POST calls after warmup, plot median, p95, distribution. For end-to-end fairness add ESP sensor sampling-to-local-alert measurement and ESP Wi-Fi HTTPS-to-remote-response measurement.
4. Offline test: disconnect Wi-Fi; measure edge works, cloud fails. Count availability not zero latency.
5. Ablation experiments: baseline fixed priority vs risk-aware priority fire response delay at controlled load. FireTask latency is measured from signal transition to buzzer pin transition using logic analyzer or timestamped ISR; current Phase 1 logs are not sufficient.
6. Training CSV has highly imbalanced 1490:10 statistical labels. Plot confusion matrix and class-specific recall; disclose class balance, labels and data leakage risks. Never claim 99% 'safety accuracy'.
