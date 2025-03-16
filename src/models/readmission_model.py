# src/models/readmission_model.py

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve
)
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
import shap
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from src.utils.feature_name_cleaner import clean_feature_names

def load_data(processed_dir):
    """
    Load the preprocessed training and testing data
    
    Parameters:
    -----------
    processed_dir : str
        Path to processed data directory
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print(f"Loading data from {processed_dir}")
    
    # Load data
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
    
    # Drop hadm_id column
    X_train = X_train.drop('hadm_id', axis=1)
    X_test = X_test.drop('hadm_id', axis=1)

    # Drop los columns due to correlation
    los_cols = [col for col in X_train.columns if 'los' in col]
    X_train = X_train.drop(los_cols, axis=1)
    X_test = X_test.drop(los_cols, axis=1)

    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Positive class (readmissions) in training: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train):.1%})")
    
    return X_train, X_test, y_train, y_test

def select_features(X_train, y_train, X_test, method='l1', threshold='median', max_features=None):
    """
    Select features using embedded methods
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : array-like
        Training target
    X_test : DataFrame
        Testing features
    method : str, default='l1'
        Feature selection method ('l1', 'rf', 'lgbm')
    threshold : str or float, default='median'
        Threshold for feature selection
    max_features : int or None, default=None
        Maximum number of features to select
        
    Returns:
    --------
    X_train_selected, X_test_selected, selected_features
    """
    print(f"Performing feature selection using {method} method...")
    
    # Choose selector based on method
    if method == 'l1':
        selector = SelectFromModel(
            LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42),
            threshold=threshold, max_features=max_features
        )
    elif method == 'rf':
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold=threshold, max_features=max_features
        )
    elif method == 'lgbm':
        selector = SelectFromModel(
            LGBMClassifier(n_estimators=100, random_state=42),
            threshold=threshold, max_features=max_features
        )
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Fit and transform
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()
    
    print(f"Selected {len(selected_features)}/{X_train.shape[1]} features")
    
    return X_train_selected, X_test_selected, selected_features

# Model training functions
def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """
    Train a logistic regression model
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    class_weight : str or dict, default='balanced'
        Class weights for imbalanced data
        
    Returns:
    --------
    Trained model
    """
    print("Training Logistic Regression model...")
    start_time = time.time()
    
    # Initialize model
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight=class_weight,
        random_state=42,
        max_iter=1000
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_random_forest(X_train, y_train, class_weight='balanced'):
    """
    Train a random forest model
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    class_weight : str or dict, default='balanced'
        Class weights for imbalanced data
        
    Returns:
    --------
    Trained model
    """
    print("Training Random Forest model...")
    start_time = time.time()
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_lightgbm(X_train, y_train):
    """Train a LightGBM classifier"""
    print("Training LightGBM model...")
    start_time = time.time()
    
    # Clean feature names
    X_train = clean_feature_names(X_train)
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        random_state=99
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    return model

def train_xgboost(X_train, y_train, scale_pos_weight=None):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    scale_pos_weight : float or None, default=None
        Weight of positive class for imbalanced data
        
    Returns:
    --------
    Trained model
    """
    print("Training XGBoost model...")
    start_time = time.time()
    
    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # Initialize model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def tune_hyperparameters(X_train, y_train, model_type='random_forest', cv=5, n_iter=20):
    """
    Tune hyperparameters for a given model type using RandomizedSearchCV
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    model_type : str, default='random_forest'
        Type of model to tune ('logistic_regression', 'random_forest', 'lightgbm', 'xgboost')
    cv : int, default=5
        Number of cross-validation folds
    n_iter : int, default=20
        Number of parameter settings to sample
        
    Returns:
    --------
    Best model
    """
    print(f"Tuning hyperparameters for {model_type}...")
    start_time = time.time()
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        param_dist = {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    
    elif model_type == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    elif model_type == 'lightgbm':
        model = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [7, 15, 31, 63],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    
    elif model_type == 'xgboost':
        # Calculate scale_pos_weight
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5, 7]
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit the search
    random_search.fit(X_train, y_train)
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best ROC AUC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

# Model evaluation functions
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model
    
    Parameters:
    -----------
    model : trained model
        Trained model to evaluate
    X_test : DataFrame or array
        Testing features
    y_test : array-like
        Testing target
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Convert to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Create dictionary of metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Plot ROC curve for a model
    
    Parameters:
    -----------
    model : trained model
        Trained model to evaluate
    X_test : DataFrame or array
        Testing features
    y_test : array-like
        Testing target
    model_name : str
        Name of the model for the plot title
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calculate AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_pr_curve(model, X_test, y_test, model_name):
    """
    Plot Precision-Recall curve for a model
    
    Parameters:
    -----------
    model : trained model
        Trained model to evaluate
    X_test : DataFrame or array
        Testing features
    y_test : array-like
        Testing target
    model_name : str
        Name of the model for the plot title
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate average precision
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', 
                label=f'Baseline (AP = {sum(y_test)/len(y_test):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plot feature importance for a model
    
    Parameters:
    -----------
    model : trained model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for the plot title
    top_n : int, default=20
        Number of top features to plot
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    # Create DataFrame of feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    return plt, feature_importance

def analyze_shap_values(model, X_test, feature_names, model_name, max_display=20, plot_type='summary', timeout=180):
    """
    Analyze SHAP values for a model with more robust error handling
    """
    print(f"Generating SHAP values for {model_name}...")
    
    try:
        # Convert X_test to DataFrame if not already
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        # Use an extremely small sample to reduce computation
        sample_size = min(100, X_test.shape[0])  # Very small sample for testing
        print(f"Using {sample_size} random samples for SHAP analysis...")
        X_sample = X_test.sample(sample_size, random_state=42)
        
        # Create figure first (in case SHAP fails)
        plt.figure(figsize=(10, 8))
        
        # Skip SHAP calculation for large models
        if X_test.shape[1] > 50 or X_test.shape[0] > 1000:
            print("Dataset too large for SHAP calculation, falling back to feature importance")
            
            # Use feature importance instead if available
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(max_display))
                plt.title(f'Feature Importance (SHAP alternative) - {model_name}')
                return plt
            
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(max_display))
                plt.title(f'Coefficient Magnitude (SHAP alternative) - {model_name}')
                return plt
            
            else:
                print("Model doesn't support feature importance, skipping SHAP analysis")
                plt.close()
                return None
        
        # Try simpler SHAP calculation approach
        try:
            print("Starting SHAP calculation with simplified approach...")
            
            # For tree models (RF, XGBoost, LightGBM)
            if hasattr(model, 'feature_importances_'):
                # Try with very small subset (10 samples)
                tiny_sample = X_sample.iloc[:10]
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(tiny_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                if plot_type == 'summary' or plot_type == 'bar':
                    shap.summary_plot(
                        shap_values, 
                        tiny_sample,
                        feature_names=tiny_sample.columns.tolist(),
                        max_display=max_display, 
                        plot_type='bar' if plot_type == 'bar' else None,
                        show=False
                    )
                
            # For linear models
            elif hasattr(model, 'coef_'):
                background = X_sample.iloc[:10].values
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(X_sample.iloc[:10].values)
                
                if plot_type == 'summary' or plot_type == 'bar':
                    shap.summary_plot(
                        shap_values, 
                        X_sample.iloc[:10],
                        feature_names=X_sample.columns.tolist(),
                        max_display=max_display, 
                        plot_type='bar' if plot_type == 'bar' else None,
                        show=False
                    )
            
            plt.title(f'SHAP Values - {model_name}')
            plt.tight_layout()
            return plt
            
        except Exception as inner_e:
            print(f"SHAP calculation failed with error: {inner_e}")
            print("Falling back to feature importance")
            
            # Use feature importance as fallback
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                fi_plot, _ = plot_feature_importance(model, feature_names, model_name, top_n=max_display)
                return fi_plot
            else:
                plt.close()
                return None
    
    except Exception as e:
        print(f"Error in SHAP analysis function: {e}")
        # Make sure any open figures are closed
        plt.close()
        return None

def save_model(model, model_path):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : trained model
        Trained model to save
    model_path : str
        Path to save the model
    """
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved to {model_path}")

def predict_with_lightgbm(model, X):
    """Make predictions with LightGBM model"""
    X = clean_feature_names(X)
    return model.predict_proba(X)[:, 1]

def find_data_directory(data_dir="data"):
    """Find the correct data directory regardless of where the script is run from"""
    # Try different possible locations
    possible_paths = [
        data_dir,                          # If run from project root
        os.path.join("..", "..", data_dir), # If run from src/models
        os.path.join("..", data_dir),      # If run from src
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", data_dir))  # Absolute path
    ]
    
    for path in possible_paths:
        processed_path = os.path.join(path, "processed")
        if os.path.exists(processed_path):
            print(f"Found data directory at: {os.path.abspath(path)}")
            return os.path.abspath(path)
    
    raise FileNotFoundError(f"Could not find data directory. Tried: {possible_paths}")

def main(data_dir="data", tune_models=False, use_feature_selection=False):
    """Main function to train and evaluate readmission prediction models"""
    # Find the correct data directory
    data_dir = find_data_directory(data_dir)
    
    # Define paths
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join(data_dir, '..', 'models')
    plots_dir = os.path.join(data_dir, '..', 'reports', 'figures')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data(processed_dir)
    
    feature_names = X_train.columns.tolist()

    # Perform feature selection if requested
    if use_feature_selection:
        X_train, X_test, selected_features = select_features(
            X_train, y_train, X_test, method='l1'
        )
        feature_names = selected_features
        print(f"Selected features: {feature_names}")
    
    # Define models to train
    models = {}
    
    # First train default models
    print("\n=== Training models with default parameters ===")
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['LightGBM'] = train_lightgbm(X_train, y_train)
    
    # Add XGBoost if data isn't too large
    if X_train.shape[0] * X_train.shape[1] < 10000000:  # Arbitrary threshold, adjust as needed
        models['XGBoost'] = train_xgboost(X_train, y_train)
    else:
        print("Skipping XGBoost due to data size")
    
    # Perform hyperparameter tuning if requested
    if tune_models:
        print("\n=== Performing hyperparameter tuning ===")
        print("This may take a while...")
        
        # Create sample for tuning if data is large
        if X_train.shape[0] > 10000:  # Arbitrary threshold, adjust as needed
            from sklearn.model_selection import train_test_split
            print("Using a subset of data for hyperparameter tuning...")
            X_tune, _, y_tune, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42, stratify=y_train)
            print(f"Tuning subset size: {X_tune.shape[0]} samples")
        else:
            X_tune, y_tune = X_train, y_train
        
        # Tune logistic regression
        print("\nTuning Logistic Regression...")
        models['Logistic Regression (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='logistic_regression')
        
        # Tune random forest
        print("\nTuning Random Forest...")
        models['Random Forest (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='random_forest')
        
        # Tune LightGBM
        print("\nTuning LightGBM...")
        models['LightGBM (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='lightgbm')
        
        # Tune XGBoost if data isn't too large
        if 'XGBoost' in models:
            print("\nTuning XGBoost...")
            models['XGBoost (Tuned)'] = tune_hyperparameters(
                X_tune, y_tune, model_type='xgboost')
    
    # Evaluate models
    metrics_list = []
    for name, model in models.items():
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics_list.append(metrics)
        
        # Plot ROC curve
        roc_plot = plot_roc_curve(model, X_test, y_test, name)
        roc_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_roc_curve.png'))
        plt.close()
        
        # Plot PR curve
        pr_plot = plot_pr_curve(model, X_test, y_test, name)
        pr_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_pr_curve.png'))
        plt.close()
        
        # Plot feature importance
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            fi_plot, fi_df = plot_feature_importance(model, feature_names, name)
            fi_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_feature_importance.png'))
            plt.close()
            
            # Save feature importance to CSV
            fi_df.to_csv(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_feature_importance.csv'), index=False)
        
        # Save model
        save_model(model, os.path.join(models_dir, f'{name.replace(" ", "_").lower()}_model.pkl'))
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame([{k: v for k, v in m.items() if k != 'confusion_matrix'} for m in metrics_list])
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(plots_dir, 'model_comparison.csv'), index=False)
    
    # Print model comparison
    print("\nModel Comparison:")
    comparison_cols = ['model_name', 'roc_auc', 'avg_precision', 'f1_score', 'precision', 'recall', 'accuracy']
    print(metrics_df[comparison_cols].sort_values('roc_auc', ascending=False))
    
    # Find best model based on ROC AUC
    best_model_idx = metrics_df['roc_auc'].argmax()
    best_model_name = metrics_df.iloc[best_model_idx]['model_name']
    print(f"\nBest model based on ROC AUC: {best_model_name}")
    
    # Analyze SHAP values for best model
    try:
        best_model = models[best_model_name]
        
        # Summary plot with 3-minute timeout
        shap_plot = analyze_shap_values(best_model, X_test, feature_names, best_model_name, 
                                       plot_type='summary', timeout=180)
        if shap_plot:
            shap_plot.savefig(os.path.join(plots_dir, f'{best_model_name.replace(" ", "_").lower()}_shap_summary.png'))
            plt.close()
        
        # Bar plot with 3-minute timeout
        shap_bar = analyze_shap_values(best_model, X_test, feature_names, best_model_name, 
                                      plot_type='bar', timeout=180)
        if shap_bar:
            shap_bar.savefig(os.path.join(plots_dir, f'{best_model_name.replace(" ", "_").lower()}_shap_bar.png'))
            plt.close()
    except Exception as e:
        print(f"Error analyzing SHAP values: {e}")
    
    print("\nReadmission model evaluation complete!")
    return metrics_df

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train readmission prediction models')
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='Directory containing processed data')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--feature_selection', action='store_true',
                        help='Perform feature selection')
    args = parser.parse_args()
    
    # Call main function
    main(args.data_dir, tune_models=args.tune, use_feature_selection=args.feature_selection)