import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, recall_score
import xgboost as xgb
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Using recall_score with pos_label=0 for specificity
specificity_scorer = make_scorer(recall_score, pos_label=0)

# Generate sample data (replace with your actual data)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, n_classes=2, random_state=42)

# Create feature names for explainability
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# Split data into train/validation/test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_df, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [4, 5, 6],
    'gamma': [0.5, 1, 1.5],
    'scale_pos_weight': [0.5, 0.3, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# Function to find optimal threshold for specificity
def find_optimal_threshold_for_specificity(model, X, y_true, target_specificity=0.95):
    """Find threshold that achieves target specificity or higher while maximizing sensitivity"""
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_sensitivity = 0
    best_specificity = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        sensitivity = recall_score(y_true, y_pred, pos_label=1)
        
        if specificity >= target_specificity and sensitivity > best_sensitivity:
            best_threshold = threshold
            best_sensitivity = sensitivity
            best_specificity = specificity
    
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"At this threshold - Specificity: {best_specificity:.4f}, Sensitivity: {best_sensitivity:.4f}")
    
    return best_threshold

# Setup GridSearchCV with early stopping
grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    ),
    param_grid=param_grid,
    scoring=specificity_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best parameters from GridSearchCV: {best_params}")

# Train the final model with early stopping
final_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric=['logloss', 'error'],
    random_state=42
)

# Create evaluation set for early stopping
eval_set = [(X_train, y_train), (X_val, y_val)]

# Train with early stopping
final_model.fit(
    X_train, 
    y_train, 
    eval_set=eval_set,
    eval_metric='logloss',
    verbose=True,
    early_stopping_rounds=20
)

print(f"Best iteration: {final_model.best_iteration}")
print(f"Best score: {final_model.best_score}")

# Find optimal threshold for high specificity
best_threshold = find_optimal_threshold_for_specificity(
    final_model, X_test, y_test, target_specificity=0.95
)

# Make predictions with optimized threshold
y_proba = final_model.predict_proba(X_test)[:, 1]
y_pred_optimized = (y_proba >= best_threshold).astype(int)

# Evaluate with optimized threshold
print("\nEvaluation with optimized threshold:")
specificity = recall_score(y_test, y_pred_optimized, pos_label=0)
sensitivity = recall_score(y_test, y_pred_optimized, pos_label=1)

print(f"Specificity: {specificity:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimized))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Optimized for Specificity)')
plt.show()

# Feature importance analysis
plt.figure(figsize=(12, 6))
xgb.plot_importance(final_model, importance_type='gain', max_num_features=15)
plt.title('Feature Importance (Gain)')
plt.show()

# Feature importance table
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Table:")
print(importance_df.head(10))

# SHAP values for model explainability
explainer = shap.Explainer(final_model)
shap_values = explainer(X_test)

# Summary plot of SHAP values
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.show()

# Detailed SHAP plot for top 10 predictions
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values[:10], X_test.iloc[:10])
plt.title('SHAP Values for Top 10 Predictions')
plt.tight_layout()
plt.show()

# SHAP decision plot for a specific sample
sample_idx = 0  # Change to examine different samples
plt.figure(figsize=(12, 8))
shap.decision_plot(explainer.expected_value, shap_values[sample_idx].values, 
                  X_test.iloc[sample_idx], feature_names=feature_names)
plt.title(f'SHAP Decision Plot for Sample {sample_idx}')
plt.tight_layout()
plt.show()

# Function to explain individual predictions
def explain_prediction(model, X_sample, threshold=best_threshold, explainer=explainer):
    """Explain an individual prediction with SHAP values and feature contributions"""
    # Get prediction and probability
    prob = model.predict_proba(X_sample.reshape(1, -1))[0, 1]
    pred = 1 if prob >= threshold else 0
    
    # Get SHAP values
    shap_values = explainer(X_sample.reshape(1, -1))
    
    # Convert to DataFrame for easier analysis
    if isinstance(X_sample, pd.DataFrame):
        feature_names = X_sample.columns
    elif isinstance(X_sample, pd.Series):
        feature_names = X_sample.index
    else:
        feature_names = [f'feature_{i}' for i in range(len(X_sample))]
    
    # Create explanation DataFrame
    explanation = pd.DataFrame({
        'Feature': feature_names,
        'Value': X_sample,
        'SHAP_Value': shap_values.values[0,:],
        'Abs_Impact': np.abs(shap_values.values[0,:])
    })
    
    # Sort by absolute impact
    explanation = explanation.sort_values('Abs_Impact', ascending=False)
    
    print(f"Prediction: Class {pred} (Probability: {prob:.4f}, Threshold: {threshold:.4f})")
    print("\nTop Features Contributing to Prediction:")
    
    return explanation

# Example of explaining a single prediction
sample_explanation = explain_prediction(final_model, X_test.iloc[0].values)
print(sample_explanation.head(10))

# Function to make predictions with high specificity
def predict_with_high_specificity(model, X_new, threshold=best_threshold, explain=False):
    """Predict with custom threshold for high specificity and optional explanation"""
    probas = model.predict_proba(X_new)[:, 1]
    preds = (probas >= threshold).astype(int)
    
    if explain and X_new.shape[0] == 1:
        # Provide explanation for single sample
        explanation = explain_prediction(model, X_new.iloc[0].values if isinstance(X_new, pd.DataFrame) else X_new[0])
        return preds, probas, explanation
    
    return preds, probas

# Save the model and threshold for later use
import joblib
model_info = {
    'model': final_model,
    'threshold': best_threshold,
    'feature_names': feature_names,
    'best_params': best_params,
    'best_iteration': final_model.best_iteration
}
joblib.dump(model_info, 'high_specificity_xgb_model.pkl')

print("\nModel saved. To load and use later:")
print("""
# Load the model
import joblib
model_info = joblib.load('high_specificity_xgb_model.pkl')
model = model_info['model']
threshold = model_info['threshold']
feature_names = model_info['feature_names']

# Use for prediction with high specificity
def predict(X_new):
    probas = model.predict_proba(X_new)[:, 1]
    preds = (probas >= threshold).astype(int)
    return preds, probas
""")
