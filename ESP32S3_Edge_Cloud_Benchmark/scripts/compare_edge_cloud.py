"""Compare same row_id predictions after you have acquired BOTH edge and cloud row-by-row predictions."""
import argparse,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--edge',required=True);p.add_argument('--cloud',required=True);p.add_argument('--out',default='results/paired_comparison.csv');a=p.parse_args()
e=pd.read_csv(a.edge).drop_duplicates('row_id');c=pd.read_csv(a.cloud).drop_duplicates('row_id')
m=e.merge(c,on='row_id',suffixes=('_edge','_cloud'))
m['prediction_equal']=m.status_edge==m.status_cloud
m.to_csv(a.out,index=False)
print('Paired:',len(m),'agreement:',m.prediction_equal.mean() if len(m) else 'NA','saved',a.out)
