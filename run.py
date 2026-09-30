"""Run Bonn path-loss experiments and print/write numeric results."""
import argparse
import csv
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import pandas as pd
from data import download, load_measurements
from experiments import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['all','controls','reference','anchors','buffers','size','gateway'],default='all')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    parser.add_argument('--protocols',nargs='+',choices=['random','packet','spatial_500m'],default=['random','packet','spatial_500m'])
    parser.add_argument('--radii',type=int,nargs='+',default=[250,500,1000])
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('results'))
    args = parser.parse_args()
    if any(r < 0 for r in args.radii):
        parser.error('Radii must be nonnegative.')
    download(args.data)
    df, meta = load_measurements(args.data/'samples.csv',args.data/'gateways.csv')
    df['gw_distance_km'] = df.gw_distance_m/1000
    print(f'{meta["used_rows"]:,} receptions; {df.gw.nunique()} gateways',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    fields = ['protocol','seed','gateway','config','model','train_rows','validation_rows',
              'test_rows','status','validation_rmse_db','test_rmse_db','test_mae_db','test_r2','selected_k']
    rows = []
    schemas = {
        'runs': ('runs.csv', fields),
        'comparison': ('comparisons.csv', ['protocol','seed','gateway','config_a','model_a',
            'config_b','model_b','grouping','point_db','lower_db','upper_db','clusters','repetitions']),
        'distance': ('distance_errors.csv', ['protocol','seed','gateway','config','model',
            'lower_km','upper_km','count','mae_db','rmse_db']),
    }
    with ExitStack() as stack:
        writers = {}
        for kind,(filename,columns) in schemas.items():
            stream = stack.enter_context((args.output/filename).open('w',newline=''))
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writers[kind] = (writer,stream)
        for record in run(df,args.study,args.seeds,args.jobs,args.protocols,args.radii):
            kind = record.pop('_kind','runs')
            if kind == 'runs':
                rows.append(record)
            writer,stream = writers[kind]
            writer.writerow(record)
            stream.flush()
    scores = pd.DataFrame(rows)
    valid = scores.loc[scores.model.ne('none')]
    if not valid.empty:
        grouped = valid.groupby(['protocol','config','model'])
        summary = grouped.test_rmse_db.agg(['mean','std','count'])
        for metric, prefix in [('test_mae_db','mae_db'),('test_r2','r2')]:
            for stat in ('mean','std','count'):
                summary[f'{prefix}_{stat}'] = grouped[metric].agg(stat)
        summary.to_csv(args.output/'summary.csv')
        print(summary.to_string(float_format=lambda x:f'{x:.4f}'))
        gateways = valid.loc[valid.protocol.eq('gateway')]
        if not gateways.empty:
            pooled=[]
            for (config,model),part in gateways.groupby(['config','model']):
                pooled.append(dict(config=config,model=model,macro_rmse_db=part.test_rmse_db.mean(),
                                   pooled_rmse_db=np.sqrt(np.average(part.test_rmse_db**2,weights=part.test_rows))))
            table = pd.DataFrame(pooled)
            table.to_csv(args.output/'gateways.csv',index=False)
            print(table.to_string(index=False))
    print(f'Output: {args.output.resolve()}')


if __name__ == '__main__':
    main()
