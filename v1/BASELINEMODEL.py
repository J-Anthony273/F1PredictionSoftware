#General imports
import os
import numpy as np
import pandas as pd

#AI model imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

#Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

fullpath = os.path.join("datasets/V1-datasets/datasetV1-filtered.csv")
df = pd.read_csv(fullpath)

features = ['FP1', 'FP2', 'FP3', 'Qualifying', 'SprintShootout', 'Sprint']
target = "Race"

df['RaceID'] = df['Season'].astype(str) + '_R' + df['RaceNumber'].astype(str)

x = df[features].copy()
y = df[target]
groups = df['RaceID']

#80/20 Train/Test split keeping races together
gss = GroupShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
train_idx, test_idx = next(gss.split(x, y, groups))

xTrain, xTest = x.iloc[train_idx], x.iloc[test_idx]
yTrain, yTest = y.iloc[train_idx], y.iloc[test_idx]
groupsTrain = groups.iloc[train_idx]
groupsTest = groups.iloc[test_idx]
metaTest = df[['Season', 'RaceNumber', 'Driver', 'Team', 'RaceID']].iloc[test_idx]

print(f"Training races : {groupsTrain.nunique()}")
print(f"Test races     : {groupsTest.nunique()}")
print(f"Training rows  : {len(xTrain)}  |  Test rows: {len(xTest)}")

#Parameter grid to determine best options for the model
paramGrid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

#Base random forest regression model
rfBase = RandomForestRegressor(random_state = 42, n_jobs = -1)

#Random forest regression model that makes use of the parameter grid and determines the best parameters
search = RandomizedSearchCV(
    estimator = rfBase,
    param_distributions = paramGrid,
    n_iter = 50,
    cv = 5,
    scoring = 'r2',
    random_state = 42,
    n_jobs = -1,
    verbose = 1
)

search.fit(xTrain, yTrain)

print(f"\nBest parameters: {search.best_params_}")
print(f"Best cross validated R2: {search.best_score_:.3f}")

rf = RandomForestRegressor(**search.best_params_, random_state = 42, n_jobs = -1)
rf.fit(xTrain, yTrain)

yPred = rf.predict(xTest)

predSeries = pd.Series(yPred, index = xTest.index)
groups  = df['RaceID'].iloc[test_idx]

yPredRanked = predSeries.copy()
for race_id, idx in groups.groupby(groups).groups.items():
    racePreds = predSeries.loc[idx]
    yPredRanked.loc[idx] = racePreds.rank(method = 'min').values

yPred = yPredRanked.values

mae = mean_absolute_error(yTest, yPred)
rmse = np.sqrt(mean_squared_error(yTest, yPred))
r2 = r2_score(yTest, yPred)

print(f"\nTest Set Results:")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

results_test = metaTest.copy()
results_test['Actual']    = yTest.values
results_test['Predicted'] = yPred

sample_race_id = results_test['RaceID'].iloc[0]

sample = results_test[results_test['RaceID'] == sample_race_id].copy()
sample = sample.sort_values('Predicted')

print(f"\n── Predictions for Race: {sample_race_id} ──")
print(f"{'Pos':<5} {'Driver':<25} {'Team':<25} {'Predicted':>10} {'Actual':>8}")
print("-" * 75)
for i, row in enumerate(sample.itertuples(), 1):
    print(f"{i:<5} {row.Driver:<25} {row.Team:<25} {row.Predicted:>10.2f} {row.Actual:>8.0f}")

race_metrics = []
for race_id, grp in results_test.groupby('RaceID'):
    actual    = grp['Actual'].values
    predicted = grp['Predicted'].values

    r_mae  = mean_absolute_error(actual, predicted)
    r_rmse = np.sqrt(mean_squared_error(actual, predicted))
    r_rho, r_p = spearmanr(actual, predicted)   

    race_metrics.append({
        'RaceID'      : race_id,
        'Season'      : grp['Season'].iloc[0],
        'RaceNumber'  : grp['RaceNumber'].iloc[0],
        'Drivers'     : len(grp),
        'MAE'         : r_mae,
        'RMSE'        : r_rmse,
        'SpearmanRho' : r_rho,
        'SpearmanP'   : r_p,
    })

race_df = pd.DataFrame(race_metrics).sort_values(['Season', 'RaceNumber'])

print(f"\n── Per-Race Summary ({'test races'}) ──")
print(race_df[['RaceID', 'Drivers', 'MAE', 'RMSE', 'SpearmanRho']].to_string(index=False))

print(f"\nMedian MAE across races  : {race_df['MAE'].median():.3f}")
print(f"Median Spearman ρ        : {race_df['SpearmanRho'].median():.3f}")
print(f"Races with ρ > 0.5       : {(race_df['SpearmanRho'] > 0.5).sum()} / {len(race_df)}")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest — F1 Race Finish Predictor (Race-Level Evaluation)',
             fontsize=14, fontweight='bold')

# Feature importances
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
colors = sns.color_palette("viridis", len(importances))
importances.plot(kind='barh', ax=axes[0, 0], color=colors)
axes[0, 0].set_title('Feature Importances', fontsize=12)
axes[0, 0].set_xlabel('Importance Score')
axes[0, 0].grid(axis='x', alpha=0.3)
for i, v in enumerate(importances):
    axes[0, 0].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

# Actual vs Predicted
axes[0, 1].scatter(yTest, yPred, alpha=0.35, color='steelblue',
                   edgecolors='none', s=18)
axes[0, 1].plot([1, 22], [1, 22], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Race Position')
axes[0, 1].set_ylabel('Predicted Race Position')
axes[0, 1].set_title('Actual vs Predicted (all test rows)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
textstr = f'MAE  = {mae:.2f}\nRMSE = {rmse:.2f}\nR²    = {r2:.3f}'
axes[0, 1].text(0.05, 0.95, textstr, transform=axes[0, 1].transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Per-race MAE distribution
axes[1, 0].hist(race_df['MAE'], bins=15, color='steelblue', edgecolor='white', alpha=0.85)
axes[1, 0].axvline(race_df['MAE'].median(), color='red', linestyle='--',
                   linewidth=1.5, label=f"Median = {race_df['MAE'].median():.2f}")
axes[1, 0].set_xlabel('MAE per Race')
axes[1, 0].set_ylabel('Number of Races')
axes[1, 0].set_title('Per-Race MAE Distribution', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Per-race Spearman rank correlation
axes[1, 1].hist(race_df['SpearmanRho'], bins=np.arange(0.0, 1.0, 0.05), color='mediumseagreen',
                edgecolor='white', alpha=0.85)
axes[1, 1].axvline(race_df['SpearmanRho'].median(), color='red', linestyle='--',
                   linewidth=1.5,
                   label=f"Median ρ = {race_df['SpearmanRho'].median():.2f}")
axes[1, 1].set_xlabel("Spearman ρ  (1 = perfect rank order)")
axes[1, 1].set_ylabel('Number of Races')
axes[1, 1].set_title('Per-Race Rank Correlation (Spearman ρ)', fontsize=12)
axes[1, 1].set_xticks(np.arange(0.0, 1.05, 0.05))
axes[1, 1].set_xlim(0.0, 1.0)     
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('graphs/rf_race_model_results.png', dpi=150, bbox_inches='tight')
plt.show()

race_df.to_csv('csvs/rf_per_race_metrics.csv', index=False)
print("\nSaved: rf_race_model_results.png  |  rf_per_race_metrics.csv")