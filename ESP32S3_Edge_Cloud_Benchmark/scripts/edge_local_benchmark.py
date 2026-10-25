"""Run same INT8 TFLite model on COMPUTER as parity reference, NOT MCU timing.
Requires full requirements.txt (tensorflow). For true MCU latency use parse_edge_serial.py.
"""
import csv,argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from server.inference import Engine
p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data/food_factory_environment_normal_abnormal.csv'));p.add_argument('--out',default=str(ROOT/'results/local_tflite_reference.csv'));p.add_argument('--mode',default='int8',choices=['int8','float32']);a=p.parse_args()
e=Engine();df=pd.read_csv(a.data);out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['row_id','label','status','score','clipped','inference_ms','mode']);w.writeheader()
 for i,r in df.reset_index(drop=True).iterrows():
  x=e.predict([r.temperature_c,r.relative_humidity_pct,r.mq135_raw],a.mode)
  w.writerow({'row_id':i,'label':r['label'],'status':x['status'],'score':x['score'],'clipped':x['input_clipped'],'inference_ms':x['inference_ms'],'mode':a.mode})
print('Saved',out,'IMPORTANT: this timing is computer CPU, not ESP32-S3.')
