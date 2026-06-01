#Imports
import os
import numpy as np
import pandas as pd
import sys

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = Tee("model_v3_output.txt")

fullpath = os.path.join("datasets/datasetV3-3.csv")
df = pd.read_csv(fullpath)

features = ['FP1', 'FP2', 'FP3', 'Qualifying', 'SprintShootout', 'Sprint',
            'driver_form', 'team_form', 'is_wet_race', 'dnf_rate',
            'carPerformance', 'driverRating', 'driverExperience',
            'driverTrackRecord', 'trackOvertakeDifficulty'
            ]
target = "Race"

df['RaceID'] = df['Season'].astype(str) + '_R' + df['RaceNumber'].astype(str)

x = df[features].copy()
y = df[target]
groups = df['RaceID']

def computeSampleWeights(dfSubset):
    weights = np.ones(len(dfSubset))
    is_dnf = dfSubset['is_dnf'].fillna(0).values
    qual = dfSubset['Qualifying'].values
    race = dfSubset['Race'].values

    dnfMask = is_dnf == 1
    prominentDNF = dnfMask & (qual <= 5) & (race >= 11)

    weights[dnfMask] = 0.3
    weights[prominentDNF] = 0.15
    return weights


gss = GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
train_idx, test_idx = next(gss.split(x, y, groups))

xTrain, xTest = x.iloc[train_idx], x.iloc[test_idx]
yTrain, yTest = y.iloc[train_idx], y.iloc[test_idx]
groupsTrain = groups.iloc[train_idx]
groupsTest = groups.iloc[test_idx]
metaTest = df[['Season', 'RaceNumber', 'Driver', 'Team', 'RaceID']].iloc[test_idx]

dfTrain = df.iloc[train_idx]
trainWeights = computeSampleWeights(dfTrain)

print(f"Training races : {groupsTrain.nunique()}")
print(f"Test races     : {groupsTest.nunique()}")
print(f"Training rows  : {len(xTrain)}  |  Test rows: {len(xTest)}")
print(f"DNF rows downweighted in training: {(trainWeights < 1).sum()} "
      f"({(trainWeights < 1).mean()*100:.1f}%)")

print("\nTuning Random Forest")

rfParamGrid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rfBase = RandomForestRegressor(random_state = 42, n_jobs = -1)

rfSearch = RandomizedSearchCV(
    estimator = rfBase,
    param_distributions = rfParamGrid,
    n_iter = 50,
    cv = 5,
    scoring = 'r2',
    random_state = 42,
    n_jobs = -1,
    verbose = 1
)

rfSearch.fit(xTrain, yTrain)

print(f"\nBest parameters: {rfSearch.best_params_}")
print(f"Best cross validated R2: {rfSearch.best_score_:.3f}")

rf = RandomForestRegressor(**rfSearch.best_params_, random_state = 42, n_jobs = -1)
rf.fit(xTrain, yTrain, sample_weight = trainWeights)

print("\nTuning XGBoost")
 
xgbParamGrid = {
    'n_estimators'  : [100, 200, 300, 500],
    'max_depth'     : [3, 5, 7, 9],
    'learning_rate' : [0.01, 0.05, 0.1, 0.2],
    'subsample'     : [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma'         : [0, 0.1, 0.3]
}

xgbBase = XGBRegressor(
    random_state = 42,
    n_jobs = -1,
    verbosity = 0,
    eval_metric = 'rmse'
)

xgbSearch = RandomizedSearchCV(
    estimator           = xgbBase,
    param_distributions = xgbParamGrid,
    n_iter              = 50,
    cv                  = 5,
    scoring             = 'r2',
    random_state        = 42,
    n_jobs              = -1,
    verbose             = 0
)

xgbSearch.fit(xTrain, yTrain, sample_weight = trainWeights)

print(f"Best XGB params : {xgbSearch.best_params_}")
print(f"Best XGB CV R²  : {xgbSearch.best_score_:.3f}")

xgb = XGBRegressor(**xgbSearch.best_params_, random_state = 42, n_jobs = -1, 
                   verbosity = 0, eval_metric = 'rmse')
xgb.fit(xTrain, yTrain, sample_weight = trainWeights)

mlpParamGrid = {
    'mlp__hidden_layer_sizes' : [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    'mlp__activation'         : ['relu', 'tanh'],
    'mlp__alpha'              : [0.0001, 0.001, 0.01, 0.1],
    'mlp__learning_rate_init' : [0.001, 0.005, 0.01],
    'mlp__max_iter'           : [500],
    'mlp__early_stopping'     : [True],
    'mlp__validation_fraction': [0.1]
}

mlpPipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp',    MLPRegressor(random_state=42))
])

mlpSearch = RandomizedSearchCV(
    estimator           = mlpPipeline,
    param_distributions = mlpParamGrid,
    n_iter              = 40,
    cv                  = 5,
    scoring             = 'r2',
    random_state        = 42,
    n_jobs              = -1,
    verbose             = 1
)

mlpSearch.fit(xTrain, yTrain)

print(f"Best MLP params : {mlpSearch.best_params_}")
print(f"Best MLP CV R²  : {mlpSearch.best_score_:.3f}")

mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        **{k.replace('mlp__', ''): v for k, v in mlpSearch.best_params_.items()},
        random_state=42
    ))
])
mlp.fit(xTrain, yTrain, mlp__sample_weight=trainWeights)

def rankPredictions(predSeries, groupSeries):
    ranked = predSeries.copy()
    for _, idx in groupSeries.groupby(groupSeries).groups.items():
        ranked.loc[idx] = predSeries.loc[idx].rank(method = 'min').values
    
    return ranked

def evaluateEnsemble(rfPred, xgbPred, mlpPred, yTrue, groupSeries, w_rf, w_xgb):
    w_mlp = round(1.0 - w_rf - w_xgb, 4)
    if w_mlp < 0:
        return None
    blended = w_rf * rfPred + w_xgb * xgbPred + w_mlp * mlpPred 
    blendSeries = pd.Series(blended, index=groupSeries.index)
    ranked      = rankPredictions(blendSeries, groupSeries)
 
    race_rhos = []
    for _, idx in groupSeries.groupby(groupSeries).groups.items():
        rho, _ = spearmanr(yTrue.loc[idx], ranked.loc[idx])
        if not np.isnan(rho):
            race_rhos.append(rho)
 
    return {
        'w_rf'      : w_rf,
        'w_xgb'     : w_xgb,
        'w_mlp'     : w_mlp,
        'median_rho': np.median(race_rhos),
        'mean_rho'  : np.mean(race_rhos),
        'rho_gt_05' : sum(r > 0.5 for r in race_rhos),
        'mae'       : mean_absolute_error(yTrue, ranked),
        'r2'        : r2_score(yTrue, ranked)
    }

print("\nSearching ensemble weights")

rfRawPred = pd.Series(rf.predict(xTest), index = xTest.index)
xgbRawPred = pd.Series(xgb.predict(xTest), index=xTest.index)
mlpRawPred = pd.Series(mlp.predict(xTest), index=xTest.index)
groupsTestSeries = groups.iloc[test_idx]

weight_results = []

for w_rf in np.round(np.arange(0.05, 0.95, 0.05), 2):
    for w_xgb in np.round(np.arange(0.05, 0.95, 0.05), 2):
        if w_rf + w_xgb >= 1.0:
            continue
        res = evaluateEnsemble(rfRawPred, xgbRawPred, mlpRawPred,
                               yTest, groupsTestSeries, w_rf, w_xgb)
        if res is None:
            continue
        weight_results.append(res)
        print(f"  RF={w_rf:.2f} / XGB={w_xgb:.2f} / MLP={res['w_mlp']:.2f}  →  "
              f"median ρ={res['median_rho']:.3f}  "
              f"MAE={res['mae']:.3f}  "
              f"R²={res['r2']:.3f}  "
              f"ρ>0.5: {res['rho_gt_05']}/{len(groupsTestSeries.unique())}")
    
weightDf = pd.DataFrame(weight_results)
bestRow = weightDf.loc[weightDf['r2'].idxmax()]
bestWRF = bestRow['w_rf']
bestWXGB = bestRow['w_xgb']
bestWMLP = bestRow['w_mlp']

print(f"\nBest weights    RF={bestWRF:.2f} / XGB={bestWXGB:.2f} / MLP={bestWMLP:.2f}")
print(f"  Median Spearman ρ : {bestRow['median_rho']:.3f}")
print(f"  MAE               : {bestRow['mae']:.3f}")
print(f"  R²                : {bestRow['r2']:.3f}")

blendedRaw  = bestWRF * rfRawPred + bestWXGB * xgbRawPred + bestWMLP * mlpRawPred
blendSeries = pd.Series(blendedRaw, index=xTest.index)
yPredRanked = rankPredictions(blendSeries, groupsTestSeries)
yPred       = yPredRanked.values
 
mae  = mean_absolute_error(yTest, yPred)
rmse = np.sqrt(mean_squared_error(yTest, yPred))
r2   = r2_score(yTest, yPred)
 
print(f"\nFinal Ensemble Test Results "
      f"(RF={bestWRF:.0%} / XGB={bestWXGB:.0%} / MLP={bestWMLP:.0%})")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
 

results_test = metaTest.copy()
results_test['Actual']    = yTest.values
results_test['Predicted'] = yPred
 
sample_race_id = results_test['RaceID'].iloc[0]
sample = results_test[results_test['RaceID'] == sample_race_id].copy()
sample = sample.sort_values('Predicted')
 
print(f"\nSample Predictions: {sample_race_id}")
print(f"{'Pos':<5} {'Driver':<25} {'Team':<25} {'Predicted':>10} {'Actual':>8}")
print("-" * 75)
for i, row in enumerate(sample.itertuples(), 1):
    flag = " DNF" if df.loc[row.Index, 'is_dnf'] == 1 else ""
    print(f"{i:<5} {row.Driver:<25} {row.Team:<25} "
          f"{row.Predicted:>10.0f} {row.Actual:>8.0f}{flag}")
 

def topNHitRate(actual, predicted, n):
    actualTopN    = set(np.where(actual    <= n)[0])
    predictedTopN = set(np.where(predicted <= n)[0])
    if len(actualTopN) == 0:
        return np.nan
    return len(actualTopN & predictedTopN) / len(actualTopN)

def withinNPositions(actual, predicted, n):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)) <= n)

def weightedSpearman(actual, predicted):
    n       = len(actual)
    weights = np.array([1 / np.log2(i + 2) for i in range(n)])
    order   = np.argsort(actual)
    w       = np.empty(n)
    w[order] = weights
    rho, _ = spearmanr(actual, predicted)
    d2     = (actual - predicted) ** 26
    wrs    = 1 - (6 * np.sum(w * d2)) / (np.sum(w) * (n ** 2 - 1))
    return wrs
 
race_metrics = []
for race_id, grp in results_test.groupby('RaceID'):
    actual    = grp['Actual'].values
    predicted = grp['Predicted'].values
    rho, rp   = spearmanr(actual, predicted)
 
    race_metrics.append({
        'RaceID'        : race_id,
        'Season'        : grp['Season'].iloc[0],
        'RaceNumber'    : grp['RaceNumber'].iloc[0],
        'Drivers'       : len(grp),
        'MAE'           : mean_absolute_error(actual, predicted),
        'RMSE'          : np.sqrt(mean_squared_error(actual, predicted)),
        'SpearmanRho'   : rho,
        'SpearmanP'     : rp,
        'WinnerCorrect' : int(predicted[np.argmin(actual)] == 1),
        'PodiumHitRate' : topNHitRate(actual, predicted, 3),
        'Top5HitRate'   : topNHitRate(actual, predicted, 5),
        'WgtSpearman'   : weightedSpearman(actual, predicted),
        'Within1Rate'   : withinNPositions(actual, predicted, 1),
        'Within3Rate'   : withinNPositions(actual, predicted, 3),
    })
 
race_df = pd.DataFrame(race_metrics).sort_values(['Season', 'RaceNumber'])

top10_results = results_test[results_test['Actual'] <= 10].copy()

top10_race_metrics = []
for race_id, grp in top10_results.groupby('RaceID'):
    actual    = grp['Actual'].values
    predicted = grp['Predicted'].values
    if len(actual) < 2:
        continue
    rho, _  = spearmanr(actual, predicted)
    top10_race_metrics.append({
        'RaceID'    : race_id,
        'Season'    : grp['Season'].iloc[0],
        'RaceNumber': grp['RaceNumber'].iloc[0],
        'MAE'       : mean_absolute_error(actual, predicted),
        'RMSE'      : np.sqrt(mean_squared_error(actual, predicted)),
        'R2'        : r2_score(actual, predicted),
        'SpearmanRho': rho,
    })

top10_race_df = pd.DataFrame(top10_race_metrics).sort_values(['Season', 'RaceNumber'])

# Overall top-10 summary metrics
mae_top10  = mean_absolute_error(top10_results['Actual'], top10_results['Predicted'])
rmse_top10 = np.sqrt(mean_squared_error(top10_results['Actual'], top10_results['Predicted']))
r2_top10   = r2_score(top10_results['Actual'], top10_results['Predicted'])
rho_top10, _ = spearmanr(top10_results['Actual'], top10_results['Predicted'])

#MLP permutation importance
print("\nComputing MLP permutation importance (may take a moment)")
mlp_perm_result = permutation_importance(
    mlp, xTest, yTest, n_repeats=10, random_state=42, n_jobs=-1
)
mlp_imp = pd.Series(
    mlp_perm_result.importances_mean, index=features
).sort_values(ascending=True)
mlp_imp_std = pd.Series(
    mlp_perm_result.importances_std, index=features
).reindex(mlp_imp.index)

print(f"\nPer-Race Summary ")
print(race_df[['RaceID', 'Drivers', 'MAE', 'RMSE', 'SpearmanRho']].to_string(index=False))
print(f"\nMedian MAE across races    : {race_df['MAE'].median():.3f}")
print(f"Median Spearman ρ          : {race_df['SpearmanRho'].median():.3f}")
print(f"Races with ρ > 0.5         : {(race_df['SpearmanRho'] > 0.5).sum()} / {len(race_df)}")
print(f"Median Weighted Spearman ρ : {race_df['WgtSpearman'].median():.3f}")
print(f"Winner predicted correctly : {race_df['WinnerCorrect'].sum()} / {len(race_df)} races "
      f"({race_df['WinnerCorrect'].mean()*100:.0f}%)")
print(f"Median podium hit rate     : {race_df['PodiumHitRate'].median()*100:.0f}%")
print(f"Median top-5 hit rate      : {race_df['Top5HitRate'].median()*100:.0f}%")
print(f"Within 1 position          : {race_df['Within1Rate'].mean()*100:.0f}%")
print(f"Within 3 positions         : {race_df['Within3Rate'].mean()*100:.0f}%")

print(f"\nTop-10 Finishers Only — Test Metrics")
print(f"MAE   (top 10): {mae_top10:.3f}")
print(f"RMSE  (top 10): {rmse_top10:.3f}")
print(f"R²    (top 10): {r2_top10:.3f}")
print(f"Spearman ρ (top 10): {rho_top10:.3f}")
print(f"Median per-race MAE  (top 10): {top10_race_df['MAE'].median():.3f}")
print(f"Median per-race RMSE (top 10): {top10_race_df['RMSE'].median():.3f}")
print(f"Median per-race R²   (top 10): {top10_race_df['R2'].median():.3f}")
print(f"Median per-race ρ    (top 10): {top10_race_df['SpearmanRho'].median():.3f}")

fig = plt.figure(figsize=(20, 24))
fig.suptitle(
    f'F1 Race Finish Predictor — Ensemble '
    f'RF({bestWRF:.0%}) + XGB({bestWXGB:.0%}) + MLP({bestWMLP:.0%})\n'
    f'DNF-Aware Training  |  Median Spearman ρ = {race_df["SpearmanRho"].median():.3f}',
    fontsize=14, fontweight='bold'
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32)

ax_fi_rf   = fig.add_subplot(gs[0, 0])
ax_fi_xgb  = fig.add_subplot(gs[0, 1])
ax_fi_mlp  = fig.add_subplot(gs[1, 0])
ax_scatter = fig.add_subplot(gs[1, 1])
ax_weight  = fig.add_subplot(gs[2, 0])
ax_mae     = fig.add_subplot(gs[2, 1])
ax_rho     = fig.add_subplot(gs[3, 0])

#Feature importance: RF
rf_imp    = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
colors_rf = sns.color_palette("viridis", len(rf_imp))
rf_imp.plot(kind='barh', ax=ax_fi_rf, color=colors_rf)
ax_fi_rf.set_title('Feature Importances — Random Forest', fontsize=11)
ax_fi_rf.set_xlabel('Importance Score')
ax_fi_rf.grid(axis='x', alpha=0.3)
for i, v in enumerate(rf_imp):
    ax_fi_rf.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

#Feature importance: XGBoost
xgb_imp    = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=True)
colors_xgb = sns.color_palette("magma", len(xgb_imp))
xgb_imp.plot(kind='barh', ax=ax_fi_xgb, color=colors_xgb)
ax_fi_xgb.set_title('Feature Importances — XGBoost', fontsize=11)
ax_fi_xgb.set_xlabel('Importance Score')
ax_fi_xgb.grid(axis='x', alpha=0.3)
for i, v in enumerate(xgb_imp):
    ax_fi_xgb.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

#Feature importance: MLP (permutation-based)
colors_mlp = sns.color_palette("crest", len(mlp_imp))
bars = ax_fi_mlp.barh(mlp_imp.index, mlp_imp.values, color=colors_mlp,
                      xerr=mlp_imp_std.values, error_kw=dict(ecolor='grey', capsize=3))
ax_fi_mlp.set_title('Feature Importances — MLP (Permutation)', fontsize=11)
ax_fi_mlp.set_xlabel('Mean Decrease in R²')
ax_fi_mlp.grid(axis='x', alpha=0.3)
for i, (v, std) in enumerate(zip(mlp_imp.values, mlp_imp_std.values)):
    ax_fi_mlp.text(v + std + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

#Actual vs Predicted scatter
ax_scatter.scatter(yTest, yPred, alpha=0.35, color='steelblue', edgecolors='none', s=18)
ax_scatter.plot([1, 22], [1, 22], 'r--', linewidth=1.5, label='Perfect Prediction')
ax_scatter.set_xlabel('Actual Race Position')
ax_scatter.set_ylabel('Predicted Race Position')
ax_scatter.set_title('Actual vs Predicted (all test rows)', fontsize=11)
ax_scatter.legend()
ax_scatter.grid(alpha=0.3)
textstr = f'MAE  = {mae:.2f}\nRMSE = {rmse:.2f}\nR²    = {r2:.3f}'
ax_scatter.text(0.05, 0.95, textstr, transform=ax_scatter.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

#Ensemble weight search
rfSummary = weightDf.groupby('w_rf')['median_rho'].max().reset_index()
ax_weight.plot(rfSummary['w_rf'], rfSummary['median_rho'],
               color='steelblue', linewidth=2, marker='o', markersize=5,
               label='Best Median ρ (per RF weight)')
ax_weight.axvline(bestWRF, color='red', linestyle='--', linewidth=1.5,
                  label=f'Best RF weight = {bestWRF:.2f}')
ax_weight.set_xlabel('Random Forest Weight')
ax_weight.set_ylabel('Median Spearman ρ')
ax_weight.set_title('Ensemble Weight Search — RF axis (best XGB+MLP per point)', fontsize=11)
ax_weight.legend(fontsize=8)
ax_weight.grid(alpha=0.3)

#Per-race MAE distribution
ax_mae.hist(race_df['MAE'], bins=15, color='steelblue', edgecolor='white', alpha=0.85)
ax_mae.axvline(race_df['MAE'].median(), color='red', linestyle='--',
               linewidth=1.5, label=f"Median = {race_df['MAE'].median():.2f}")
ax_mae.set_xlabel('MAE per Race')
ax_mae.set_ylabel('Number of Races')
ax_mae.set_title('Per-Race MAE Distribution', fontsize=11)
ax_mae.legend()
ax_mae.grid(axis='y', alpha=0.3)

#Per-race Spearman ρ distribution
ax_rho.hist(race_df['SpearmanRho'], bins=np.arange(0.0, 1.05, 0.05),
            color='mediumseagreen', edgecolor='white', alpha=0.85)
ax_rho.axvline(race_df['SpearmanRho'].median(), color='red', linestyle='--',
               linewidth=1.5, label=f"Median ρ = {race_df['SpearmanRho'].median():.2f}")
ax_rho.set_xlabel("Spearman ρ  (1 = perfect rank order)")
ax_rho.set_ylabel('Number of Races')
ax_rho.set_title('Per-Race Rank Correlation (Spearman ρ)', fontsize=11)
ax_rho.set_xticks(np.arange(0.0, 1.05, 0.1))
ax_rho.set_xlim(0.0, 1.0)
ax_rho.legend()
ax_rho.grid(axis='y', alpha=0.3)

# Hide unused bottom-right cell
fig.add_subplot(gs[3, 1]).set_visible(False)

plt.savefig('graphs/rf_race_model_results2.png', dpi=150, bbox_inches='tight')
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle(
    f'Top-10 Finishers Only — Per-Race Metric Distributions\n'
    f'Overall: MAE={mae_top10:.2f}  RMSE={rmse_top10:.2f}  '
    f'R²={r2_top10:.3f}  Spearman ρ={rho_top10:.3f}',
    fontsize=13, fontweight='bold'
)

#MAE — top 10
ax_t10_mae = axes2[0, 0]
ax_t10_mae.hist(top10_race_df['MAE'], bins=15, color='steelblue',
                edgecolor='white', alpha=0.85)
ax_t10_mae.axvline(top10_race_df['MAE'].median(), color='red', linestyle='--',
                   linewidth=1.5,
                   label=f"Median = {top10_race_df['MAE'].median():.2f}")
ax_t10_mae.set_xlabel('MAE (top-10 drivers per race)')
ax_t10_mae.set_ylabel('Number of Races')
ax_t10_mae.set_title('Per-Race MAE — Top 10 Only', fontsize=11)
ax_t10_mae.legend()
ax_t10_mae.grid(axis='y', alpha=0.3)

#RMSE — top 10
ax_t10_rmse = axes2[0, 1]
ax_t10_rmse.hist(top10_race_df['RMSE'], bins=15, color='darkorange',
                 edgecolor='white', alpha=0.85)
ax_t10_rmse.axvline(top10_race_df['RMSE'].median(), color='red', linestyle='--',
                    linewidth=1.5,
                    label=f"Median = {top10_race_df['RMSE'].median():.2f}")
ax_t10_rmse.set_xlabel('RMSE (top-10 drivers per race)')
ax_t10_rmse.set_ylabel('Number of Races')
ax_t10_rmse.set_title('Per-Race RMSE — Top 10 Only', fontsize=11)
ax_t10_rmse.legend()
ax_t10_rmse.grid(axis='y', alpha=0.3)

#R² — top 10
ax_t10_r2 = axes2[1, 0]
ax_t10_r2.hist(top10_race_df['R2'], bins=15, color='mediumpurple',
               edgecolor='white', alpha=0.85)
ax_t10_r2.axvline(top10_race_df['R2'].median(), color='red', linestyle='--',
                  linewidth=1.5,
                  label=f"Median = {top10_race_df['R2'].median():.3f}")
ax_t10_r2.set_xlabel('R² (top-10 drivers per race)')
ax_t10_r2.set_ylabel('Number of Races')
ax_t10_r2.set_title('Per-Race R² — Top 10 Only', fontsize=11)
ax_t10_r2.legend()
ax_t10_r2.grid(axis='y', alpha=0.3)

#Spearman ρ — top 10
ax_t10_rho = axes2[1, 1]
ax_t10_rho.hist(top10_race_df['SpearmanRho'],
                bins=np.arange(-1.0, 1.05, 0.1),
                color='mediumseagreen', edgecolor='white', alpha=0.85)
ax_t10_rho.axvline(top10_race_df['SpearmanRho'].median(), color='red',
                   linestyle='--', linewidth=1.5,
                   label=f"Median ρ = {top10_race_df['SpearmanRho'].median():.3f}")
ax_t10_rho.set_xlabel("Spearman ρ (top-10 drivers per race)")
ax_t10_rho.set_ylabel('Number of Races')
ax_t10_rho.set_title('Per-Race Spearman ρ — Top 10 Only', fontsize=11)
ax_t10_rho.set_xlim(-1.0, 1.0)
ax_t10_rho.legend()
ax_t10_rho.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('graphs/rf_race_model_results_top10.png', dpi=150, bbox_inches='tight')
plt.show()

race_df.to_csv('csvs/rf_per_race_metrics2.csv', index=False)
top10_race_df.to_csv('csvs/rf_per_race_metrics_top10.csv', index=False)
weightDf.to_csv('csvs/ensemble_weight_search2.csv', index=False)

predictionsDF = results_test[['Season', 'RaceNumber', 'Driver', 'Actual', 'Predicted']].copy()
predictionsDF['Difference'] = (predictionsDF['Predicted'] - predictionsDF['Actual']).abs()
predictionsDF = predictionsDF.sort_values(['Season', 'RaceNumber', 'Actual']).reset_index(drop = True)

predictionsDF.to_csv('csvs/driverPredictions.csv', index = False)


print("\nSaved: rf_race_model_results2.png | rf_race_model_results_top10.png")
print("rf_per_race_metrics2.csv | rf_per_race_metrics_top10.csv | ensemble_weight_search2.csv")

sys.stdout.file.close()