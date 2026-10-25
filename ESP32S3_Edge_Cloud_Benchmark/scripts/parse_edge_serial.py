"""Extract real ESP32-S3 Invoke() latencies from ESP-IDF logs.
Supports Phase 1 and Full Safety Demo logs. Does not infer accuracy from status.
"""
import argparse,csv,re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--out',default='results/edge_serial.csv')
a=p.parse_args()
output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
# Both "latency=62us" and "latency=62 us" are valid.
latency=re.compile(r'latency\s*=\s*(\d+)\s*us\b',re.I)
score=re.compile(r'\bscore\s*=\s*([\d.eE+-]+)',re.I)
with output.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['sample_id','inference_ms','score','raw_log']);w.writeheader()
 n=0
 for line in Path(a.input).open(errors='replace'):
  if not (('SAFETY_DEMO:' in line and ('AI score=' in line or 'REAL AI:' in line)) or ('ENV_RTOS:' in line and 'AI:' in line)):continue
  m=latency.search(line)
  if not m:continue
  sc=score.search(line)
  w.writerow({'sample_id':n,'inference_ms':int(m.group(1))/1000,'score':sc.group(1) if sc else '', 'raw_log':line.strip()})
  n+=1
print('Actual ESP32 hardware Invoke() samples:',n)
if not n:print('No valid model Invoke() measurements found. Input clipping prevents an Invoke() measurement. Do NOT use fire-rule display as AI latency.')
print('Output:',output)
