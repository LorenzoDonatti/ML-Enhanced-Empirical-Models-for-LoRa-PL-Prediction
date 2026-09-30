"""Numeric experiments: baselines, anchors, matched inputs, buffers and gateways."""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import split_rows, packet_groups, fit_ldpl_bonn_train, CLASSICAL_BASELINES, spatial_groups
from models import fit_config, rmse, metrics

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


def paired_bootstrap(y, a, b, groups, repetitions=2000):
    """Paired cluster percentile CI for RMSE(a) - RMSE(b), conditional on fits."""
    codes, unique = pd.factorize(groups, sort=False)
    counts = np.bincount(codes)
    a_sse = np.bincount(codes, weights=(y-a)**2)
    b_sse = np.bincount(codes, weights=(y-b)**2)
    rng = np.random.default_rng(0)
    delta = np.empty(repetitions)
    for i in range(repetitions):
        selected = rng.integers(0, len(unique), len(unique))
        n = counts[selected].sum()
        delta[i] = np.sqrt(a_sse[selected].sum()/n)-np.sqrt(b_sse[selected].sum()/n)
    return dict(point_db=rmse(y,a)-rmse(y,b),
                lower_db=float(np.quantile(delta,.025)), upper_db=float(np.quantile(delta,.975)),
                clusters=len(unique), repetitions=repetitions)


def diagnostics(df, test, predictions, protocol, seed, gateway):
    """Seed-zero distance bins and paper comparisons, without storing raw predictions."""
    if seed != 0:
        return
    y = df.path_loss.to_numpy()[test]
    common = dict(protocol=protocol, seed=seed, gateway=gateway)
    bins = np.floor(df.gw_distance_m.to_numpy()[test]/500).astype(int)
    for (config, model), prediction in predictions.items():
        for bin_id in np.unique(bins):
            selected = bins == bin_id
            scores = metrics(y[selected], prediction[selected])
            yield dict(_kind='distance', **common, config=config, model=model,
                       lower_km=bin_id*.5, upper_km=(bin_id+1)*.5, count=int(selected.sum()),
                       mae_db=scores['test_mae_db'], rmse_db=scores['test_rmse_db'])
    if protocol not in ('random','packet','spatial_500m') and not protocol.startswith('buffer'):
        return
    grouping = 'packet' if protocol in ('random','packet') else 'spatial_500m'
    groups = (packet_groups(df) if grouping == 'packet' else spatial_groups(df)).iloc[test]
    pairs = [(('oh_basic','ensemble'),('oh_basic',m)) for m in ('rf','xgboost','knn')]
    pairs += [((a,'ensemble'),(b,'ensemble')) for a,b in
              [('oh_height','direct_height'),('oh_basic','direct_height'),('oh_basic','direct_basic')]]
    for a,b in pairs:
        if a in predictions and b in predictions:
            yield dict(_kind='comparison', **common, config_a=a[0], model_a=a[1],
                       config_b=b[0], model_b=b[1], grouping=grouping,
                       **paired_bootstrap(y,predictions[a],predictions[b],groups))


def evaluate(df, parts, configs, seed, jobs, protocol, basics=False, gateway=''):
    train, val, test = parts
    common = dict(protocol=protocol, seed=seed, gateway=gateway,
                  train_rows=len(train), validation_rows=len(val), test_rows=len(test))
    if len(train) < 100 or len(val) < 100:
        yield {**common, 'config':'infeasible', 'model':'none', 'status':'infeasible'}
        return
    df['ldpl_bonn_train'], _ = fit_ldpl_bonn_train(df, train)
    y = df.path_loss.to_numpy()
    predictions = {}
    mean_prediction = np.full(len(test), y[train].mean())
    yield {**common, 'config':'training_mean', 'model':'constant', **metrics(y[test],mean_prediction)}
    if seed == 0:
        predictions[('training_mean','constant')] = mean_prediction
    if basics or protocol == 'gateway':
        for name in ([*CLASSICAL_BASELINES, 'ldpl_bonn_train'] if basics else ['okumura_hata']):
            prediction = df[name].to_numpy()[test]
            yield {**common, 'config':name, 'model':'classical', **metrics(y[test],prediction)}
            if seed == 0:
                predictions[(name,'classical')] = prediction
    if basics:
        x = df[['latitude','longitude','log_distance_km']].to_numpy()
        for name, base in [('direct_basic', np.zeros(len(df))),
                           ('oh_basic', df.okumura_hata.to_numpy())]:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(x[train], y[train]-base[train])
            prediction = base[test]+model.predict(x[test])
            yield {**common, 'config':name, 'model':'linear', **metrics(y[test],prediction)}
            if seed == 0:
                predictions[(name,'linear')] = prediction
    for config in configs:
        records, fitted_predictions = fit_config(df, parts, config, seed, jobs)
        print(f'{protocol} seed={seed} {gateway} {config}: '
              f'{records[-1]["test_rmse_db"]:.4f} dB', flush=True)
        for record in records:
            yield {**common, 'config':config, **record}
        if seed == 0:
            predictions.update({(config,m):p for m,p in fitted_predictions.items()})
    yield from diagnostics(df,test,predictions,protocol,seed,gateway)


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
