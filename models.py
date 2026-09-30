"""RF, XGBoost, validation-tuned KNN and equal direct/residual ensembles."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

CONFIGS = {
    'direct_basic': ('zero', []),
    'direct_height': ('zero', ['antenna_height']),
    'direct_anchor': ('zero', ['okumura_hata']),
    'oh_basic': ('okumura_hata', []),
    'oh_height': ('okumura_hata', ['antenna_height']),
    'ldpl_basic': ('ldpl_bonn_train', []),
    'oh_continuous': ('okumura_hata', []),
    'oh_distance_only': ('okumura_hata', []),
    'fspl_residual': ('fspl', []),
    'oulu_residual': ('ldpl_oulu', []),
    'winner_residual': ('winnerplus', []),
}

def make_model(name: str, seed: int, jobs: int):
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=jobs)
    if name == "xgboost":
        return XGBRegressor(n_estimators=100, random_state=seed, n_jobs=jobs,
                            objective="reg:squarederror")
    raise ValueError(f"Unknown model: {name}")

def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, prediction)))

def metrics(y, prediction):
    """Metrics on final path-loss predictions; R² is undefined for constant targets."""
    r2 = float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else float('nan')
    return {'test_rmse_db': rmse(y, prediction),
            'test_mae_db': float(mean_absolute_error(y, prediction)), 'test_r2': r2}

def fit_config(df, parts, config, seed, jobs):
    train, validation, test = parts
    anchor, extra = CONFIGS[config]
    y = df.path_loss.to_numpy()
    base = np.zeros(len(df)) if anchor == 'zero' else df[anchor].to_numpy()
    target = y - base
    tree_distance = 'gw_distance_km' if config == 'oh_continuous' else 'gw_distance_rounded_km'
    tree_features = ['latitude', 'longitude', tree_distance] + extra
    knn_features = ['latitude', 'longitude', 'log_distance_km'] + extra
    if config == 'oh_distance_only':
        tree_features = [tree_distance]
        knn_features = ['log_distance_km']
    predictions, records = {}, []
    x_tree = df[tree_features].to_numpy()
    x_knn = df[knn_features].to_numpy()
    for algorithm in ('rf', 'xgboost', 'knn'):
        if algorithm == 'knn':
            choices = []
            for k in (5, 10, 20, 50, 100):
                if k > len(train):
                    continue
                model = make_pipeline(StandardScaler(), KNeighborsRegressor(
                    n_neighbors=k, weights='distance', n_jobs=jobs))
                model.fit(x_knn[train], target[train])
                val_pred = base[validation] + model.predict(x_knn[validation])
                choices.append((rmse(y[validation], val_pred), k, model))
            val_rmse, chosen_k, model = min(choices, key=lambda x: (x[0], x[1]))
            x = x_knn
        else:
            chosen_k = None
            x = x_tree
            model = make_model(algorithm, seed, jobs)
            model.fit(x[train], target[train])
            val_rmse = rmse(y[validation], base[validation] + model.predict(x[validation]))
        prediction = base[test] + model.predict(x[test])
        predictions[algorithm] = prediction
        records.append({'model': algorithm, 'validation_rmse_db': val_rmse,
                        **metrics(y[test], prediction),
                        'selected_k': chosen_k})
    predictions['ensemble'] = np.mean(list(predictions.values()), axis=0)
    records.append({'model': 'ensemble', **metrics(y[test], predictions['ensemble'])})
    return records, predictions
