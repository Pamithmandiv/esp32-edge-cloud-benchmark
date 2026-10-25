"""Create figures + JSON metrics from observed CSV. Never invent latency/accuracy."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,balanced_accuracy_score,precision_recall_fscore_support
P=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--cloud',required=True);a.add_argument('--edge',default='');a.add_argument('--out',default=str(P/'results/graphs'));o=a.parse_args()
out=Path(o.out);out.mkdir(exist_ok=True,parents=True)
cloud=pd.read_csv(o.cloud);edge=pd.read_csv(o.edge) if o.edge else None
metrics={}
def lat(df,col):
 x=pd.to_numeric(df[col],errors='coerce').dropna().to_numpy()
 return {'n':len(x),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'mean_ms':float(x.mean())} if len(x) else None
for col in ('client_http_ms','server_total_ms','inference_ms'):
 if col in cloud:metrics['cloud_'+col]=lat(cloud,col)
if edge is not None and 'inference_ms' in edge:metrics['edge_inference_ms']=lat(edge,'inference_ms')
# Accuracy uses dataset's experimental proxy labels only, exclude clipped/error, dedupe repeats.
for name,df in [('cloud',cloud),('edge',edge)]:
 if df is None or not {'label','status','row_id'}.issubset(df.columns):continue
 clean=df[df['label'].astype(str).str.lower().isin(['normal','abnormal']) & df['status'].isin(['NORMAL','ABNORMAL'])].drop_duplicates('row_id')
 if len(clean):
  true=(clean.label.astype(str).str.lower()=='abnormal').astype(int)
  pred=(clean.status=='ABNORMAL').astype(int)
  tn,fp,fn,tp=confusion_matrix(true,pred,labels=[0,1]).ravel()
  pr,rc,f1,_=precision_recall_fscore_support(true,pred,average='binary',zero_division=0)
  metrics[name+'_proxy_classification']={'n_evaluated':int(len(clean)),'n_excluded':int(len(df.drop_duplicates('row_id'))-len(clean)), 'accuracy':float(accuracy_score(true,pred)),'balanced_accuracy':float(balanced_accuracy_score(true,pred)),'precision_abnormal':float(pr),'recall_abnormal':float(rc),'f1_abnormal':float(f1),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
# Graph 1 comparative median; raw categories are intentionally separate: chip vs server vs HTTP
vals=[]
for title,key in [('Edge ESP32 inference','edge_inference_ms'),('Cloud server inference','cloud_inference_ms'),('Cloud HTTP round trip','cloud_client_http_ms')]:
 if metrics.get(key):vals.append((title,metrics[key]['median_ms']))
if vals:
 fig,ax=plt.subplots(figsize=(9,4.5));ax.bar([x[0] for x in vals],[x[1] for x in vals]);ax.set_ylabel('Median latency (milliseconds)');ax.set_title('Measured latency (different measurement boundaries)');plt.xticks(rotation=12,ha='right');fig.tight_layout();fig.savefig(out/'01_latency_median.png',dpi=170);plt.close(fig)
# Graph 2 actual HTTP distribution
x=pd.to_numeric(cloud.get('client_http_ms',pd.Series(dtype=float)),errors='coerce').dropna()
if len(x):
 fig,ax=plt.subplots(figsize=(8,4.5));ax.hist(x,bins=min(30,max(5,int(len(x)**.5))));ax.set_xlabel('HTTP round-trip ms');ax.set_ylabel('Requests');ax.set_title('Cloud end-to-end round-trip latency');fig.tight_layout();fig.savefig(out/'02_cloud_http_distribution.png',dpi=170);plt.close(fig)
# Graph 3 benchmark proxy label accuracy
names=[];accuracies=[]
for name in ('edge','cloud'):
 m=metrics.get(name+'_proxy_classification')
 if m:names.append(name.title());accuracies.append(m['accuracy']*100)
if names:
 fig,ax=plt.subplots(figsize=(6,4));ax.bar(names,accuracies);ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title('Experimental proxy-label accuracy (not safety accuracy)');fig.tight_layout();fig.savefig(out/'03_proxy_accuracy.png',dpi=170);plt.close(fig)
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2));print('Saved figures in',out)
