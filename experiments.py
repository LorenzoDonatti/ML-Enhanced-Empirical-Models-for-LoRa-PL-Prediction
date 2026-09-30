"""Numeric experiments: baselines, anchors, matched inputs, buffers and gateways."""
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import split_rows, packet_groups, fit_ldpl_bonn_train, CLASSICAL_BASELINES
from models import fit_config, rmse

RADIUS_M = 6371008.8

def minimum_distances(coordinates, query, reference):
    if not len(query):
        return np.empty(0)
    if not len(reference):
        return np.full(len(query), np.inf)
    tree = BallTree(coordinates[reference], metric='haversine')
    return tree.query(coordinates[query], k=1)[0][:, 0] * RADIUS_M

def buffered_split(df, seed, buffer_m):
    """Preserve original test tiles; discard nearby development observations.

    Retain validation rows outside the test buffer, then training rows outside
    both test and retained validation buffers. Test rows never change with radius.
    Distances refer to transmitter positions, including across gateways.
    """
    train, validation, test = split_rows(df, 'spatial_500m', seed)
    coordinates = np.deg2rad(df[['latitude', 'longitude']].to_numpy())
    if buffer_m > 0:
        validation = validation[minimum_distances(coordinates, validation, test) >= buffer_m]
        train = train[minimum_distances(coordinates, train, test) >= buffer_m]
        train = train[minimum_distances(coordinates, train, validation) >= buffer_m]
    details = {'train': len(train), 'validation': len(validation), 'test': len(test),
               'buffer_m': buffer_m,
               'min_train_test_m': float(minimum_distances(coordinates, test, train).min()),
               'min_validation_test_m': float(minimum_distances(coordinates, validation, test).min()) if len(validation) else None,
               'min_train_validation_m': float(minimum_distances(coordinates, validation, train).min()) if len(validation) else None}
    return (train, validation, test), details


def evaluate(df, parts, configs, seed, jobs, protocol, basics=False, gateway=''):
    train, val, test = parts
    common = dict(protocol=protocol, seed=seed, gateway=gateway,
                  train_rows=len(train), validation_rows=len(val), test_rows=len(test))
    if len(train) < 100 or len(val) < 100:
        yield {**common, 'config':'infeasible', 'model':'none', 'status':'infeasible'}
        return
    df['ldpl_bonn_train'], _ = fit_ldpl_bonn_train(df, train)
    y = df.path_loss.to_numpy()
    if basics:
        for name in [*CLASSICAL_BASELINES, 'ldpl_bonn_train']:
            yield {**common, 'config':name, 'model':'classical',
                   'test_rmse_db':rmse(y[test], df[name].to_numpy()[test])}
        x = df[['latitude','longitude','log_distance_km']].to_numpy()
        for name, base in [('direct_basic', np.zeros(len(df))),
                           ('oh_basic', df.okumura_hata.to_numpy())]:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(x[train], y[train]-base[train])
            yield {**common, 'config':name, 'model':'linear',
                   'test_rmse_db':rmse(y[test], base[test]+model.predict(x[test]))}
    for config in configs:
        records, _ = fit_config(df, parts, config, seed, jobs)
        print(f'{protocol} seed={seed} {gateway} {config}: '
              f'{records[-1]["test_rmse_db"]:.4f} dB', flush=True)
        for record in records:
            yield {**common, 'config':config, **record}


def run(df, study, seeds, jobs, protocols, radii):
    controls = ['direct_basic','direct_height','direct_anchor','oh_basic',
                'oh_height','ldpl_basic','oh_continuous']
    anchors = ['fspl_residual','oulu_residual','ldpl_basic','oh_basic','winner_residual']
    for protocol in protocols:
        for seed in seeds:
            if study in ('all', 'controls', 'reference'):
                configs = controls if study in ('all','controls') else ['oh_basic','oh_distance_only']
                if study == 'all':
                    configs = list(dict.fromkeys(configs+anchors+['oh_distance_only']))
                yield from evaluate(df, split_rows(df,protocol,seed), configs,
                                    seed,jobs,protocol, basics=study in ('all','reference'))
            elif study == 'anchors':
                yield from evaluate(df,split_rows(df,protocol,seed),anchors,
                                    seed,jobs,protocol,basics=True)
    if study in ('all','buffers','size'):
        for radius in radii:
            for seed in seeds:
                parts, details = buffered_split(df,seed,radius)
                if study in ('all','buffers'):
                    yield from evaluate(df,parts,['direct_height','direct_anchor','oh_basic','oh_height'],
                                        seed,jobs,f'buffer{radius}')
                if study in ('all','size'):
                    train,val,test = split_rows(df,'spatial_500m',seed)
                    rng = np.random.default_rng(seed)
                    train = np.sort(rng.choice(train,details['train'],replace=False))
                    val = np.sort(rng.choice(val,details['validation'],replace=False))
                    yield from evaluate(df,(train,val,test),['direct_height','oh_height'],
                                        seed,jobs,f'size{radius}')
    if study in ('all','gateway'):
        indices = np.arange(len(df))
        groups = packet_groups(df)
        for gateway in sorted(df.gw.unique()):
            test = indices[df.gw.eq(gateway)]
            remaining = indices[df.gw.ne(gateway)]
            splitter = GroupShuffleSplit(n_splits=1,test_size=.15,random_state=0)
            tr,va = next(splitter.split(remaining,groups=groups.iloc[remaining]))
            yield from evaluate(df,(remaining[tr],remaining[va],test),
                                ['direct_height','oh_height','oh_basic'],0,jobs,'gateway',gateway=gateway)
