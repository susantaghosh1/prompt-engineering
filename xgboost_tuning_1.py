"""learning_rate (eta): Controls the contribution of each tree. Lower values (0.01-0.3) make the model more robust but require more trees.
max_depth: Controls tree complexity. Values of 3-6 often work well; deeper trees can lead to overfitting.
subsample: The fraction of samples used per tree (0.5-1.0). Values < 1.0 help prevent overfitting.
colsample_bytree: The fraction of features used per tree. Values < 1.0 add randomness and help prevent overfitting.
n_estimators: The number of trees. Often needs tuning in conjunction with learning_rate.
min_child_weight: Minimum sum of instance weight needed in a child node. Higher values prevent overfitting.
gamma: Minimum loss reduction required for a split. Higher values make the algorithm more conservative.

Sequential Tuning: Often more effective to tune in groups rather than all at once:

First tune tree-specific parameters (max_depth, min_child_weight)
Then tune randomness parameters (subsample, colsample_bytree)
Finally tune learning_rate with n_estimators
"""
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
    'scale_pos_weight': [1, (y_train==0).sum() / (y_train==1).sum()]  # For imbalanced datasets
}
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500],  # Set this higher than needed
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

# Create the base XGBClassifier without eval_metric
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False  # Avoid warning for newer XGBoost versions
)

# The fit_params will be passed to the fit method of the estimator
fit_params = {
    "eval_set": [(X_val, y_val)],  # Separate validation set
    "eval_metric": ["auc", "logloss"],  # Metrics specified here in fit_params
    "early_stopping_rounds": 20,  # Stop if no improvement after 20 rounds
    "verbose": False
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',  # Metric for selecting best model in CV
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Pass the fit parameters through the fit method
grid_search.fit(X_train, y_train, **fit_params)

print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Print metrics
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance (optional)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save the best model (optional)
best_model.save_model('best_xgboost_model.json')

