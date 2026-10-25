# ESP32-S3 Edge vs Google Cloud Run: Quickstart

## Important scientific boundaries
The INT8 .tflite in this package is byte-for-byte identical to the 3,544-byte model embedded in the user's uploaded ESP32-S3 demo firmware (SHA256: 9df4c7122d948bf34a3e6ebdd7f41a519020805a9f4b4e23ecbac0b379e44843).
The ESP32 OLED "MODEL: NORMAL/ABNORMAL" from DEMO_VIEW_MODE is fire-rule simulation, **NOT** a TensorFlow Lite prediction. Do not use that display rule for model accuracy. Its real DHT22 inputs (e.g. RH 80%) cause INT8 clipping and inference rejection, so the current demo firmware alone produces **zero valid ESP32 Invoke() measurements** in that state. Edge latency requires real valid Invoke() log lines, or a deliberately labeled reference-vector replay mode on the microcontroller. Never fabricate these.

## Cloud service run locally
Use Python 3.11 (or Docker) and project root. Full server requirements include tensorflow-cpu (can be large).
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 127.0.0.1 --port 8080
```
In another terminal: `curl http://127.0.0.1:8080/health`. API docs: http://127.0.0.1:8080/docs

## Deploy same application to Google Cloud Run
- Sign in to https://console.cloud.google.com/ , create/select project, enable billing and install gcloud CLI locally or use Cloud Shell with uploaded project folder.
- From this project directory after `gcloud auth login`:
```
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy safety-cloud-int8 --source . --region asia-south1 --allow-unauthenticated --memory 2Gi --cpu 1 --concurrency 1 --max-instances 1
```
- Cloud Run requires a billing-enabled project. Actual billing depends on usage. Public endpoint is only suitable for synthetic/non-sensitive benchmark vectors. For real industrial data restrict authorization and avoid exposing it publicly.
- Set a Google Cloud Budget/Alerts in Billing. Region availability can vary; change region if necessary.
- Dockerfile binds `0.0.0.0` and uses Cloud Run injected `$PORT`.

## Measure Cloud latency on same dataset
Open a new terminal in project root. For client scripts, a lightweight env can use requirements-client.txt. Otherwise existing full env is fine.
```
python scripts/benchmark.py --url https://YOUR_SERVICE_URL --limit 30 --out results/cloud_30.csv
python scripts/graph_only_cloud.py --csv results/cloud_30.csv
```
Then all rows and repeat runs:
```
python scripts/benchmark.py --url https://YOUR_SERVICE_URL --repeat 3 --out results/cloud_int8.csv
python scripts/graph_only_cloud.py --csv results/cloud_int8.csv
```
Outputs: per-request `inference_ms` (server-side TFLite), `server_total_ms` (server processing), `client_http_ms` (client observed HTTPS round trip, includes network). For meaningful measurements run benchmark **from your Sri Lankan Mac** rather than from Cloud Shell; Cloud Shell latency is from Google's network, NOT Sri Lankan client network.

## Measure ESP32 hardware inference latency
Use **actual** ESP-IDF logs with `REAL AI: NORMAL ... latency=NN us` or earlier `AI: ... latency=NNus`. Current hardware demo may only log `Actual AI: INVALID - input clipping`, yielding zero valid samples. Do not read OLED demo status as real AI.
For ESP-IDF monitor save logs:
```
idf.py -p /dev/cu.YOUR_PORT monitor 2>&1 | tee results/esp32_monitor.txt
```
Close monitor with Ctrl+]. Then:
```
python scripts/parse_edge_serial.py --input results/esp32_monitor.txt --out results/edge_serial.csv
python scripts/graph_only_edge.py --csv results/edge_serial.csv
```
For firmware that has no valid invokes, add a separately labeled reference-vector replay benchmark to ESP32 `ai_task`, run the exact model and known normal-range test vectors, log every Invoke() us. Do not treat this test as real sensor-to-alert latency or live deployment performance.

## Graph comparing both
After BOTH genuine CSV files exist:
```
python scripts/graphs.py --cloud results/cloud_int8.csv --edge results/edge_serial.csv
```
Outputs PNG figures in results/graphs and metrics.json. Median/P95 are better than cherry-picked minima. Device latency vs server inference compares compute runtimes; device Invoke() vs cloud HTTP RTT has different measurement boundaries, so clearly disclose that. Best edge use-case argument is **no connectivity requirement and local alert availability**; edge is not necessarily faster per-op or more accurate than cloud.

## Accuracy caveat
The bundled dataset has experimental proxy labels and was used for model development. Class imbalance can inflate accuracy. An **honest** edge accuracy comparison requires device inference outputs for the identical held-out labeled vectors, and labels independently verified; the demo's fire-rule `MODEL: ABNORMAL` is not valid classifier output. Cloud accuracy graph based on proxy labels must be marked as development/experimental only.
