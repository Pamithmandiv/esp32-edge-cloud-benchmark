# ESP32-S3 Edge vs Cloud — FAIR comparison research kit

**Same pretrained artifact. No retraining necessary for INT8-to-INT8 comparison.** Included actual user model weights (INT8 3,544 bytes) and original labeled CSV. Cloud mode defaults to SAME INT8 TFLite model/threshold/normalization as ESP32 edge. Optional Float32 endpoint uses same trained model before quantization but does **not** use INT8 threshold or claim classification accuracy: float model threshold must be re-calibrated on normal validation only.

## Research design
- Edge: ESP32-S3 does local inference without network; your Phase1 log showed inference tens of microseconds, not end-to-end acquisition or alert latency.
- Cloud: FastAPI server on Google Cloud Run performs model `Invoke()` on server after sensor fields arrive in HTTPS POST.
- **Inference-only latency** measured independently on each CPU; **cloud end-to-end HTTP round trip** timed on client (includes networking, serialization, queuing, inference). **Edge end-to-end** requires measuring sensor-to-alert on-device, not comparing a microsecond `Invoke()` measurement with cloud RTT as if equivalent.
- **Accuracy** needs held-out human/independently verified ground truth. Included CSV 1490 normal vs 10 abnormal has statistical proxy labels and *was involved in the earlier model development*. Full 1500-row accuracy can have leakage and 99.3% majority baseline. Use original untouched time-split test plus verified hazard challenge rows. No claimed worker-safety validation.
- MQ135 = simulated 70–77 for user's demo: do not report simulated data as real measurements. To compare modalities feed *identical* [temperature,humidity,MQ135] triples to both branches. Dataset is best first comparison; live paired ESP tests need firmware extension to read/replay same row_id vectors and log edge predictions and HTTP client RTT (or separate Wi-Fi task sending samples, never blocking safety task).

## Files
- `server/app.py` cloud HTTP API /predict and /health
- `server/inference.py` reads exact INT8 model and its preprocessing, rejects clipped samples with `INPUT_OUT_OF_RANGE`
- `model/` original user models/config/test vectors
- `data/food_factory_environment_normal_abnormal.csv` original dataset (statistically created labels)
- `scripts/benchmark.py` cloud benchmark using same dataset rows, writes observed CSV
- `scripts/edge_local_benchmark.py` Python desktop TFLite parity check — NOT ESP32 latency
- `scripts/parse_edge_serial.py` real MCU inference latency from Phase1 serial log
- `scripts/graphs.py` three PNG figures and metrics JSON, no invented results
- `scripts/compare_edge_cloud.py` paired row-by-row prediction agreement. Requires actual edge row-wise file with row_id.

## Cloud hosting recommendation
**Google Cloud Run** (serverless Docker). Choose an Asia region near Sri Lanka e.g. `asia-south1` if supported for your account. Deployment needs cloud project, billing enabled, `gcloud` CLI; costs possible. Set budget alert and delete service when finished. Alternative Render free Web Service may cold-start after 15 min idle, distorting latency.

### A: Local cloud-server prototype on Mac (computer acts as server)
Python 3.11 recommended, Docker alternative below:
```bash
cd ESP32S3_Edge_Cloud_Benchmark
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8080
```
Second terminal:
```bash
source .venv/bin/activate
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/predict -H 'Content-Type: application/json' -d '{"temperature_c":27.5,"relative_humidity_pct":41.0,"mq135_raw":74,"mode":"int8","source":"SIMULATED_DEMO"}'
python scripts/benchmark.py --url http://127.0.0.1:8080 --limit 30
```
This measures LAN/local server, **not genuine remote cloud latency**.

### B: Deploy **real remote cloud** to Google Cloud Run
1. Install Google Cloud CLI, login, select project with billing enabled, API permissions.
2. In this extracted folder run:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy food-factory-ai --source . --region asia-south1 --allow-unauthenticated --memory 2Gi --cpu 1 --concurrency 1 --max-instances 1
```
The command builds provided Dockerfile from source and returns service URL. Change region if not offered. Service exposed publicly by command: use only for synthetic demonstration data, never confidential workplace readings. To require an API key, set Cloud Run env `API_KEY` and send matching `X-API-Key` header (better: Cloud Run IAM authentication / private endpoint for real data). Read billing prompts and quotas; billing is not automatically free.
3. Verify `https://YOUR_URL/health`, then POST to `https://YOUR_URL/predict`.
4. Run benchmark **from Sri Lanka laptop on stable Wi-Fi** (same connection for tests):
```bash
pip install -r requirements-client.txt
python scripts/benchmark.py --url https://YOUR_URL --out results/cloud_int8.csv --repeat 3
python scripts/graphs.py --cloud results/cloud_int8.csv
```
If API_KEY set, export API_KEY in client environment. First Cloud Run request can cold start; script warms 3 requests, so chart depicts WARM performance; cold-start experiment separate.

### C: Edge measured inference timings from actual ESP-IDF log
Save real monitor log (it has lines `AI: ... latency=62us`) to `results/esp_monitor.txt`, then:
```bash
python scripts/parse_edge_serial.py --input results/esp_monitor.txt --out results/edge_serial.csv
python scripts/graphs.py --cloud results/cloud_int8.csv --edge results/edge_serial.csv
```
This gives a **valid hardware latency distribution** but `edge_serial.csv` has NO ground-truth labels and Phase 1 contains only the same hardcoded vector; no edge accuracy claim possible. To make a true paired, labeled edge accuracy comparison, flash a vector replay version of firmware, load labeled held-out vectors and get `row_id,label,prediction`; original model on local desktop is only a parity check and must NOT be relabeled as MCU results.

### D: Optional same-model local TFLite reference (not hardware)
```bash
python scripts/edge_local_benchmark.py --out results/local_tflite_reference.csv
python scripts/graphs.py --cloud results/cloud_int8.csv --edge results/local_tflite_reference.csv
```
This gives reference model predictions; the timing is from **Mac**, not ESP32. Keep figures titled accordingly. Use `--mode float32` for numeric scores only; float32 classification threshold is not calibrated.

### E: Important fairness and accuracy notes
- Dataset full CSV used for dev/calibration, so measure on a held-out never-used label-verified test set for publishable accuracy. 1490 normal, 10 abnormal → accuracy alone misleading; show confusion matrix, abnormal recall, false-positive rate and balanced accuracy. Model might incorrectly label some clip/unseen inputs. `INPUT_OUT_OF_RANGE` is NOT NORMAL.
- Compare same samples and same quantization config for Edge-vs-Cloud architecture. Cloud Float32 option is a **different model representation** so treat its results separately (quantization study).
- Cloud-only safety alerts should fail safe on network outage; Edge remains local. Fire sensor is **not model inference** and should be an independent high-priority device task, not submitted to cloud.
- `MQ135=70..77` is synthetic demonstration value. Do not include simulated data in real-world accuracy claims.
- Avoid using private human camera footage on a publicly accessible endpoint.
- ESP serial timestamps are not cloud RTT. For an actual ESP-cloud comparison you need add ESP Wi-Fi HTTPS client; current kit provides backend and bench scripts, not a modified integrated on-device HTTP firmware.

### Cleanup after demo
```bash
gcloud run services delete food-factory-ai --region asia-south1
```
Review Cloud Build / Artifact Registry storage in billing as well.
