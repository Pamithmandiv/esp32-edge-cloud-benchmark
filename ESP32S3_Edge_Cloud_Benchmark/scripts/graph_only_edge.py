"""Plot genuine ESP32 inference latencies, not display-rule updates."""
import argparse
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',default='results/graphs/edge_latency.png');args=p.parse_args()
df=pd.read_csv(args.csv)
x=pd.to_numeric(df['inference_ms'],errors='coerce').dropna().to_numpy()
if not len(x):raise SystemExit('No actual ESP32 inference latency measurements; need actual valid Invoke() calls.')
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(9,4.5))
ax.plot(np.arange(1,len(x)+1),x,marker='.',linewidth=1)
ax.axhline(np.median(x),linestyle='--',label=f'Median {np.median(x):.4f} ms')
ax.set(xlabel='ESP32 inference sample number',ylabel='Invoke-only latency (ms)',title='Measured ESP32-S3 inference latency')
ax.legend();fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)
print('N:',len(x),'median ms:',np.median(x),'P95 ms:',np.percentile(x,95),'chart:',out)
