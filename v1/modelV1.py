#Imports
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

fullpath = os.path.join("datasets/V1-datasets/datasetV1-filtered.csv")
df = pd.read_csv(fullpath)

features = ['FP1', 'FP2', 'FP3', 'Qualifying', 'SprintShootout', 'Sprint']
target = "Race"

x = df[features].copy()
y = df[target]

labels = df[['Driver', 'Team']].reset_index(drop = True)

xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size = 0.2, random_state = 42
)

paramGrid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rfBase = RandomForestRegressor(random_state = 42, n_jobs = -1)

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
mae = mean_absolute_error(yTest, yPred)
rmse = np.sqrt(mean_squared_error(yTest, yPred))
r2 = r2_score(yTest, yPred)

print(f"\nTest Set Results:")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Random Forest Regression — F1 Race Finish Predictor',
             fontsize=15, fontweight='bold')

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
colors = sns.color_palette("viridis", len(importances))
importances.plot(kind='barh', ax=axes[0], color=colors)
axes[0].set_title('Feature Importances', fontsize=13)
axes[0].set_xlabel('Importance Score')
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(importances):
    axes[0].text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

# Actual vs Predicted
axes[1].scatter(yTest, yPred, alpha=0.35, color='steelblue', edgecolors='none', s=18)
axes[1].plot([1, 22], [1, 22], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[1].set_xlabel('Actual Race Position')
axes[1].set_ylabel('Predicted Race Position')
axes[1].set_title('Actual vs Predicted Race Position', fontsize=13)
axes[1].legend()
axes[1].grid(alpha=0.3)
textstr = f'MAE  = {mae:.2f}\nRMSE = {rmse:.2f}\nR²    = {r2:.3f}'
axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('graphs/rf_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
