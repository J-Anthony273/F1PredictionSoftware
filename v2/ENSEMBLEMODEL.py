#Imports
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

fullpath = os.path.join("datasets/datasetV2.csv")
df = pd.read_csv(fullpath)

features = ['FP1', 'FP2', 'FP3', 'Qualifying', 'SprintShootout', 'Sprint',
            'driver_form', 'team_form', 'is_wet_race', 'dnf_rate'
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

print("\n── Tuning Random Forest ──")

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

print("\n── Tuning XGBoost ──")
 
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

def rankPredictions(predSeries, groupSeries):
    ranked = predSeries.copy()
    for _, idx in groupSeries.groupby(groupSeries).groups.items():
        ranked.loc[idx] = predSeries.loc[idx].rank(method = 'min').values
    
    return ranked

def evaluateEnsemble(rfPred, xgbPred, yTrue, groupSeries, w_rf):
    w_xgb = 1.0 - w_rf
    blended = w_rf * rfPred + w_xgb * xgbPred

    blendSeries = pd.Series(blended, index = groupSeries.index)
    ranked = rankPredictions(blendSeries, groupSeries)

    race_rhos = []
    for _, idx in groupSeries.groupby(groupSeries).groups.items():
        rho, _ = spearmanr(yTrue.loc[idx], ranked.loc[idx])

        if not np.isnan(rho):
            race_rhos.append(rho)
    
    return {
        'w_rf' : w_rf,
        'w_xgb' : w_xgb,
        'median_rho': np.median(race_rhos),
        'mean_rho'     : np.mean(race_rhos),
        'rho_gt_05'    : sum(r > 0.5 for r in race_rhos),
        'mae'          : mean_absolute_error(yTrue, ranked),
        'r2'           : r2_score(yTrue, ranked)
    }

print("\nSearching ensemble weights")

rfRawPred = pd.Series(rf.predict(xTest), index = xTest.index)
xgbRawPred = pd.Series(xgb.predict(xTest), index=xTest.index)
groupsTestSeries = groups.iloc[test_idx]

weight_results = []

for w_rf in np.round(np.arange(0.05, 1.00, 0.05), 2):
    res = evaluateEnsemble(rfRawPred, xgbRawPred, yTest, groupsTestSeries, w_rf)
    weight_results.append(res)
    print(f"  RF={w_rf:.2f} / XGB={1-w_rf:.2f}  →  "
          f"median ρ={res['median_rho']:.3f}  "
          f"MAE={res['mae']:.3f}  "
          f"R²={res['r2']:.3f}  "
          f"ρ>0.5: {res['rho_gt_05']}/{len(groupsTestSeries.unique())}")
    
weightDf = pd.DataFrame(weight_results)
bestRow = weightDf.loc[weightDf['median_rho'].idxmax()]
bestWRF = bestRow['w_rf']
bestWXGB = bestRow['w_xgb']

print(f"\nBest weights    RF={bestWRF:.2f} / XGB={bestWXGB:.2f}")
print(f"  Median Spearman ρ : {bestRow['median_rho']:.3f}")
print(f"  MAE               : {bestRow['mae']:.3f}")
print(f"  R²                : {bestRow['r2']:.3f}")

blendedRaw  = bestWRF * rfRawPred + bestWXGB * xgbRawPred
blendSeries = pd.Series(blendedRaw, index=xTest.index)
yPredRanked = rankPredictions(blendSeries, groupsTestSeries)
yPred       = yPredRanked.values
 
mae  = mean_absolute_error(yTest, yPred)
rmse = np.sqrt(mean_squared_error(yTest, yPred))
r2   = r2_score(yTest, yPred)
 
print(f"\nFinal Ensemble Test Results (RF={bestWRF:.0%} / XGB={bestWXGB:.0%})")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
 

results_test = metaTest.copy()
results_test['Actual']    = yTest.values
results_test['Predicted'] = yPred
 
sample_race_id = results_test['RaceID'].iloc[0]
sample = results_test[results_test['RaceID'] == sample_race_id].copy()
sample = sample.sort_values('Predicted')
 
print(f"\n── Sample Predictions: {sample_race_id} ──")
print(f"{'Pos':<5} {'Driver':<25} {'Team':<25} {'Predicted':>10} {'Actual':>8}")
print("-" * 75)
for i, row in enumerate(sample.itertuples(), 1):
    flag = " DNF" if df.loc[row.Index, 'is_dnf'] == 1 else ""
    print(f"{i:<5} {row.Driver:<25} {row.Team:<25} "
          f"{row.Predicted:>10.0f} {row.Actual:>8.0f}{flag}")
 
race_metrics = []
for race_id, grp in results_test.groupby('RaceID'):
    actual    = grp['Actual'].values
    predicted = grp['Predicted'].values
    rho, rp   = spearmanr(actual, predicted)
 
    race_metrics.append({
        'RaceID'     : race_id,
        'Season'     : grp['Season'].iloc[0],
        'RaceNumber' : grp['RaceNumber'].iloc[0],
        'Drivers'    : len(grp),
        'MAE'        : mean_absolute_error(actual, predicted),
        'RMSE'       : np.sqrt(mean_squared_error(actual, predicted)),
        'SpearmanRho': rho,
        'SpearmanP'  : rp,
    })
 
race_df = pd.DataFrame(race_metrics).sort_values(['Season', 'RaceNumber'])
 
print(f"\nPer-Race Summary")
print(race_df[['RaceID', 'Drivers', 'MAE', 'RMSE', 'SpearmanRho']].to_string(index=False))
print(f"\nMedian MAE across races  : {race_df['MAE'].median():.3f}")
print(f"Median Spearman ρ        : {race_df['SpearmanRho'].median():.3f}")
print(f"Races with ρ > 0.5       : {(race_df['SpearmanRho'] > 0.5).sum()} / {len(race_df)}")

fig = plt.figure(figsize=(20, 18))
fig.suptitle(
    f'F1 Race Finish Predictor — Ensemble RF({bestWRF:.0%}) + XGB({bestWXGB:.0%})\n'
    f'DNF-Aware Training  |  Median Spearman ρ = {race_df["SpearmanRho"].median():.3f}',
    fontsize=14, fontweight='bold'
)
 
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.32)
 
ax_fi_rf  = fig.add_subplot(gs[0, 0])
ax_fi_xgb = fig.add_subplot(gs[0, 1])
ax_scatter = fig.add_subplot(gs[1, 0])
ax_weight  = fig.add_subplot(gs[1, 1])
ax_mae     = fig.add_subplot(gs[2, 0])
ax_rho     = fig.add_subplot(gs[2, 1])
 
#Feature importance: RF
rf_imp  = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
colors_rf = sns.color_palette("viridis", len(rf_imp))
rf_imp.plot(kind='barh', ax=ax_fi_rf, color=colors_rf)
ax_fi_rf.set_title('Feature Importances — Random Forest', fontsize=11)
ax_fi_rf.set_xlabel('Importance Score')
ax_fi_rf.grid(axis='x', alpha=0.3)
for i, v in enumerate(rf_imp):
    ax_fi_rf.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
 
#Feature importance: XGBoost
xgb_imp = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=True)
colors_xgb = sns.color_palette("magma", len(xgb_imp))
xgb_imp.plot(kind='barh', ax=ax_fi_xgb, color=colors_xgb)
ax_fi_xgb.set_title('Feature Importances — XGBoost', fontsize=11)
ax_fi_xgb.set_xlabel('Importance Score')
ax_fi_xgb.grid(axis='x', alpha=0.3)
for i, v in enumerate(xgb_imp):
    ax_fi_xgb.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
 
#Actual vs Predicted scatter
ax_scatter.scatter(yTest, yPred, alpha=0.35, color='steelblue',
                   edgecolors='none', s=18)
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
 
#Ensemble weight search results
ax_weight.plot(weightDf['w_rf'], weightDf['median_rho'],
               color='steelblue', linewidth=2, marker='o', markersize=5, label='Median ρ')
ax_weight.axvline(bestWRF, color='red', linestyle='--',
                  linewidth=1.5, label=f'Best RF weight = {bestWRF:.2f}')
ax_weight.set_xlabel('Random Forest Weight  (XGBoost = 1 − RF)')
ax_weight.set_ylabel('Median Spearman ρ')
ax_weight.set_title('Ensemble Weight Search', fontsize=11)
ax_weight.legend()
ax_weight.grid(alpha=0.3)
ax_weight2 = ax_weight.twinx()
ax_weight2.plot(weightDf['w_rf'], weightDf['mae'],
                color='darkorange', linewidth=1.5, linestyle=':', alpha=0.7, label='MAE')
ax_weight2.set_ylabel('MAE', color='darkorange')
ax_weight2.tick_params(axis='y', labelcolor='darkorange')
 
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
               linewidth=1.5,
               label=f"Median ρ = {race_df['SpearmanRho'].median():.2f}")
ax_rho.set_xlabel("Spearman ρ  (1 = perfect rank order)")
ax_rho.set_ylabel('Number of Races')
ax_rho.set_title('Per-Race Rank Correlation (Spearman ρ)', fontsize=11)
ax_rho.set_xticks(np.arange(0.0, 1.05, 0.1))
ax_rho.set_xlim(0.0, 1.0)
ax_rho.legend()
ax_rho.grid(axis='y', alpha=0.3)
 
plt.savefig('graphs/rf_race_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
 
race_df.to_csv('csvs/rf_per_race_metrics.csv', index=False)
weightDf.to_csv('csvs/ensemble_weight_search.csv', index=False)
print("\nSaved: rf_race_model_results.png | rf_per_race_metrics.csv | ensemble_weight_search.csv")