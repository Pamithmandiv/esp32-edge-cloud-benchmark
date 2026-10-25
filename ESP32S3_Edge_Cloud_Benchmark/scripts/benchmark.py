"""Measure cloud end-to-end HTTP and server inference for the SAME labeled rows.
No measured numbers are fabricated. Predictions exclude clipped rows from accuracy.
"""
import argparse,csv,os,time,sys,json,statistics
from pathlib import Path
import requests,pandas as pd
BASE=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',required=True,help='https://SERVICE.run.app or http://127.0.0.1:8080')
    ap.add_argument('--data',default=str(BASE/'data/food_factory_environment_normal_abnormal.csv'))
    ap.add_argument('--out',default=str(BASE/'results/cloud_results.csv'))
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--repeat',type=int,default=1)
    ap.add_argument('--mode',choices=['int8','float32'],default='int8')
    ap.add_argument('--source',choices=['DATASET','SIMULATED_DEMO','REAL'],default='DATASET')
    args=ap.parse_args()
    data=pd.read_csv(args.data)
    if args.limit:data=data.iloc[:args.limit]
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    s=requests.Session();h={'X-API-Key':os.getenv('API_KEY','')} if os.getenv('API_KEY') else {}
    endpoint=args.url.rstrip('/')+'/predict'
    # warm up and record no warm-up latency; sleep-based cold starts must be separate
    sample=data.iloc[0]
    payload={'temperature_c':float(sample.temperature_c),'relative_humidity_pct':float(sample.relative_humidity_pct),'mq135_raw':float(sample.mq135_raw),'mode':args.mode,'source':args.source}
    for _ in range(3):s.post(endpoint,json=payload,headers=h,timeout=90).raise_for_status()
    cols=['row_id','repeat_id','label','temperature_c','relative_humidity_pct','mq135_raw','source','status','score','threshold','clipped','inference_ms','server_total_ms','client_http_ms','http_status','error']
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for rep in range(args.repeat):
            for ix,(_,row) in enumerate(data.iterrows()):
                p={'temperature_c':float(row.temperature_c),'relative_humidity_pct':float(row.relative_humidity_pct),'mq135_raw':float(row.mq135_raw),'mode':args.mode,'source':args.source}
                start=time.perf_counter_ns();res=None;error='';status_code=0
                try:
                    res=s.post(endpoint,json=p,headers=h,timeout=30);status_code=res.status_code;res.raise_for_status();v=res.json()
                except requests.RequestException as ex: error=str(ex);v={}
                ms=(time.perf_counter_ns()-start)/1e6
                w.writerow({'row_id':ix,'repeat_id':rep,'label':row.get('label',''),'temperature_c':p['temperature_c'],'relative_humidity_pct':p['relative_humidity_pct'],'mq135_raw':p['mq135_raw'],'source':args.source,'status':v.get('status','ERROR'),'score':v.get('score',''),'threshold':v.get('threshold',''),'clipped':v.get('input_clipped',''), 'inference_ms':v.get('inference_ms',''), 'server_total_ms':v.get('server_total_ms',''),'client_http_ms':ms,'http_status':status_code,'error':error})
                if ix%100==0:print('repeat',rep,'row',ix,'http_ms',round(ms,2),'status',v.get('status',error[:60]),flush=True)
    print('Results:',out)
if __name__=='__main__':main()
