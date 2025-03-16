# src/models/resource_model.py

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from lightgbm import LGBMRegressor
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from src.utils.feature_name_cleaner import clean_feature_names

def load_data(processed_dir, include_readmission_pred=True):
    """
    Load the preprocessed training and testing data
    
    Parameters:
    -----------
    processed_dir : str
        Path to processed data directory
    include_readmission_pred : bool, default=True
        Whether to include readmission predictions as a feature
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, feature_names, target_col
    """
    print(f"Loading data from {processed_dir}")
    
    target_col = 'icu_los_days'

    # Load the data from hf_data_processed.csv
    data_path = os.path.join(processed_dir, 'hf_data_processed.csv')
    data = pd.read_csv(data_path)

    # Split into features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Drop ID column as it should not be used as a feature
    print("Dropping hadm_id column from features...")
    X_train = X_train.drop(columns=['hadm_id'], errors='ignore')
    X_test = X_test.drop(columns=['hadm_id'], errors='ignore')

    # Drop total_los_days if present (too closely related to target)
    if 'total_los_days' in X_train.columns:
        print("Dropping total_los_days column from features...")
        X_train = X_train.drop(columns=['total_los_days'])
        X_test = X_test.drop(columns=['total_los_days'])

    # Convert y to a 1D array
    y_train = y_train.values
    y_test = y_test.values

    # Include readmission predictions if requested
    if include_readmission_pred:
        # Load the best readmission model
        models_dir = os.path.join(os.path.dirname(processed_dir), '..', 'models')
        
        try:
            # Try to find the best model based on convention
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            
            if model_files:
                # Use the first available model
                model_path = os.path.join(models_dir, model_files[0])
                print(f"Loading readmission model from {model_path}")
                
                with open(model_path, 'rb') as f:
                    readmission_model = pickle.load(f)
                
                # Generate readmission predictions
                readmission_proba_train = readmission_model.predict_proba(X_train)[:, 1]
                readmission_proba_test = readmission_model.predict_proba(X_test)[:, 1]
                
                # Add to features
                X_train = X_train.copy()
                X_test = X_test.copy()
                X_train['readmission_probability'] = readmission_proba_train
                X_test['readmission_probability'] = readmission_proba_test
                
                print("Added readmission probability as a feature")
            else:
                print("No readmission model found. Proceeding without readmission predictions.")
        except Exception as e:
            print(f"Error loading readmission model: {e}")
            print("Proceeding without readmission predictions.")
    


    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Target variable: {target_col} (mean value in training: {np.mean(y_train):.2f})")

    return X_train, X_test, y_train, y_test, X_train.columns.tolist(), target_col

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    Trained model
    """
    print("Training Linear Regression model...")
    start_time = time.time()
    
    # Initialize model
    model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train a ridge regression model
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    alpha : float, default=1.0
        Regularization strength
        
    Returns:
    --------
    Trained model
    """
    print("Training Ridge Regression model...")
    start_time = time.time()
    
    # Initialize model
    model = Ridge(alpha=alpha, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_elastic_net(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    """
    Train an elastic net regression model
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter (0 <= l1_ratio <= 1)
        l1_ratio=0 corresponds to Ridge, l1_ratio=1 to Lasso
        
    Returns:
    --------
    Trained model
    """
    print("Training Elastic Net model...")
    start_time = time.time()
    
    # Initialize model
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        max_iter=1000
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_random_forest_regressor(X_train, y_train):
    """
    Train a random forest regressor
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    Trained model
    """
    print("Training Random Forest Regressor...")
    start_time = time.time()
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_lightgbm_regressor(X_train, y_train):
    """Train a LightGBM regressor"""
    print("Training LightGBM Regressor...")
    start_time = time.time()
    
    # Clean feature names before training
    X_train_clean = clean_feature_names(X_train)
    
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_clean, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def train_quantile_regressor(X_train, y_train, quantiles=[0.5, 0.75, 0.9]):
    """
    Train quantile regression models for different percentiles
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    quantiles : list, default=[0.5, 0.75, 0.9]
        List of quantiles to predict
        
    Returns:
    --------
    Dictionary of trained models, keyed by quantile
    """
    print("Training Quantile Regression models...")
    start_time = time.time()
    
    # Initialize dict to store models
    models = {}
    
    # Train a model for each quantile
    for q in quantiles:
        print(f"  Training quantile {q:.2f} model...")
        q_start_time = time.time()
        
        # Initialize model
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=q,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store model
        models[f'q{int(q*100)}'] = model
        
        # Print training time
        q_elapsed_time = time.time() - q_start_time
        print(f"  Quantile {q:.2f} model completed in {q_elapsed_time:.2f} seconds")
    
    # Print total training time
    elapsed_time = time.time() - start_time
    print(f"All quantile models completed in {elapsed_time:.2f} seconds")
    
    return models

def train_mlp_regressor(X_train, y_train):
    """
    Train a neural network regressor
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    Trained model
    """
    print("Training Neural Network (MLP) Regressor...")
    start_time = time.time()
    
    # Initialize model
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Print training time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return model

def tune_hyperparameters(X_train, y_train, model_type='ridge', cv=5, n_iter=20):
    """
    Tune hyperparameters for a given model type using RandomizedSearchCV
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : array-like
        Training target
    model_type : str, default='ridge'
        Type of model to tune ('ridge', 'elastic_net', 'random_forest', 'lightgbm', 'mlp')
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
    
    if model_type == 'ridge':
        model = Ridge(random_state=42)
        param_dist = {
            'alpha': np.logspace(-3, 3, 7)
        }
    
    elif model_type == 'elastic_net':
        model = ElasticNet(random_state=42, max_iter=1000)
        param_dist = {
            'alpha': np.logspace(-3, 3, 7),
            'l1_ratio': np.linspace(0.1, 0.9, 9)
        }
    
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    elif model_type == 'lightgbm':
        model = LGBMRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [7, 15, 31, 63],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    
    elif model_type == 'mlp':
        model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
        param_dist = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': np.logspace(-5, -3, 3),
            'learning_rate': ['constant', 'adaptive'],
            'batch_size': [32, 64, 128, 'auto']
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
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
    print(f"Best RMSE: {np.sqrt(-random_search.best_score_):.4f}")
    
    return random_search.best_estimator_

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
    
    # Handle quantile regression models differently
    if isinstance(model, dict) and list(model.keys())[0].startswith('q'):
        # Evaluate each quantile model and return the median (q50) for comparison
        metrics_list = []
        
        # Make predictions for each quantile
        predictions = {}
        for q_name, q_model in model.items():
            predictions[q_name] = q_model.predict(X_test)
        
        # Use the median (q50) model for standard metrics if available
        if 'q50' in model:
            y_pred = predictions['q50']
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE, handling zeros appropriately
            try:
                y_test_nonzero = np.where(y_test == 0, 1e-10, y_test)
                mape = np.mean(np.abs((y_test - y_pred) / y_test_nonzero)) * 100
            except:
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            print(f"Median (q50) model metrics:")
            print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"  Mean Absolute Error (MAE): {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            
            # Create dictionary of metrics for the median model
            metrics = {
                'model_name': model_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        else:
            # If q50 not available, use the first quantile for reporting
            first_q = list(model.keys())[0]
            y_pred = predictions[first_q]
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            try:
                y_test_nonzero = np.where(y_test == 0, 1e-10, y_test)
                mape = np.mean(np.abs((y_test - y_pred) / y_test_nonzero)) * 100
            except:
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            print(f"Note: No q50 model found. Using {first_q} for metrics.")
            print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"  Mean Absolute Error (MAE): {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            
            metrics = {
                'model_name': model_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        
        # Add quantile coverage metrics
        for q_name, q_model in model.items():
            if q_name.startswith('q') and q_name != 'q50':
                q_value = float(q_name[1:]) / 100
                y_pred_q = predictions[q_name]
                coverage = np.mean(y_test <= y_pred_q)
                print(f"  {q_name} coverage: {coverage:.4f} (target: {q_value:.2f})")
                metrics[f'{q_name}_coverage'] = coverage
        
        return metrics
    
    else:
        # Standard model evaluation
        # Make predictions
        if 'LightGBM' in model_name:
            X_test_clean = clean_feature_names(X_test)
            y_pred = model.predict(X_test_clean)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE, handling zeros appropriately
        try:
            # Replace zeros with small value to avoid division by zero
            y_test_nonzero = np.where(y_test == 0, 1e-10, y_test)
            mape = np.mean(np.abs((y_test - y_pred) / y_test_nonzero)) * 100
        except:
            # Fallback to scikit-learn's implementation
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Print metrics
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Create dictionary of metrics
        metrics = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        return metrics

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    y_test : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Add metrics to plot
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    plt.text(
        0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    return plt

def plot_quantile_predictions(y_test, model_dict, X_test, model_name):
    """
    Plot quantile predictions
    
    Parameters:
    -----------
    y_test : array-like
        Actual values
    model_dict : dict
        Dictionary of quantile models
    X_test : DataFrame or array
        Test features
    model_name : str
        Name of the model for the plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Sort samples by actual value for better visualization
    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    
    # Calculate and sort predictions for each quantile
    predictions = {}
    for q_name, q_model in model_dict.items():
        predictions[q_name] = q_model.predict(X_test)[sorted_indices]
    
    # Plot the actual values
    plt.plot(range(len(y_test_sorted)), y_test_sorted, 'k-', label='Actual')
    
    # Plot each quantile
    colors = ['b', 'g', 'r']
    color_idx = 0
    
    # Sort quantiles for consistent plotting
    sorted_q_names = sorted(model_dict.keys())
    
    for q_name in sorted_q_names:
        q_value = float(q_name[1:]) / 100
        plt.plot(
            range(len(y_test_sorted)), 
            predictions[q_name], 
            f'{colors[color_idx % len(colors)]}-', 
            alpha=0.7, 
            label=f'{q_name} ({q_value:.2f})'
        )
        color_idx += 1
    
    plt.xlabel('Samples (sorted by actual value)')
    plt.ylabel('Value')
    plt.title(f'Quantile Predictions - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

def plot_residuals(y_test, y_pred, model_name):
    """
    Plot residuals (actual - predicted)
    
    Parameters:
    -----------
    y_test : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for the plot title
    """
    residuals = y_test - y_pred
    
    plt.figure(figsize=(14, 6))
    
    # Residuals vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Analysis - {model_name}')
    plt.tight_layout()
    
    return plt

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plot feature importance for a trained model
    """
    plt.figure(figsize=(10, 8))
    
    if hasattr(model, 'feature_importances_'):
        # Get raw importance values
        importances = model.feature_importances_
        
        # Normalize to [0,1] range if it's a LightGBM model
        if model_name.lower().startswith('lightgbm'):
            importances = importances / importances.sum()
            print("Normalized LightGBM feature importances to sum to 1.0")
        
        # Create DataFrame for sorting and plotting
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Print the sum of feature importance
        print(f"Sum of feature importances: {importances.sum():.4f}")
        
        # Plot
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        
        return plt, feature_importance
    else:
        print(f"Model {model_name} doesn't support feature_importances_ attribute")
        return None, None

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

def predict_resource_needs(model, X_data, model_name, target_col):
    """
    Generate resource predictions and basic statistics
    
    Parameters:
    -----------
    model : trained model
        Best trained model
    X_data : DataFrame
        Features to predict on
    model_name : str
        Name of the model
    target_col : str
        Name of the target column
        
    Returns:
    --------
    DataFrame with predictions and summary statistics
    """
    # Handle quantile model dictionaries differently
    if isinstance(model, dict) and list(model.keys())[0].startswith('q'):
        print("Generating predictions from quantile regression models...")
        
        # Create a DataFrame to store all predictions
        results = X_data.copy()
        
        # Generate predictions for each quantile
        for q_name, q_model in model.items():
            results[f'predicted_{target_col}_{q_name}'] = q_model.predict(X_data)
        
        # Use median (q50) for summary if available
        if 'q50' in model:
            predictions = results[f'predicted_{target_col}_q50']
            median_label = 'q50'
        else:
            # Use first quantile as fallback
            first_q = list(model.keys())[0]
            predictions = results[f'predicted_{target_col}_{first_q}']
            median_label = first_q
        
        # Calculate summary statistics
        summary = {
            'model_name': f"{model_name} ({median_label})",
            'target': target_col,
            'mean_prediction': np.mean(predictions),
            'median_prediction': np.median(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'total_resource_needs': np.sum(predictions)
        }
        
        # Add quantile-specific statistics
        for q_name in model.keys():
            q_predictions = results[f'predicted_{target_col}_{q_name}']
            q_value = float(q_name[1:]) / 100
            
            summary[f'{q_name}_mean'] = np.mean(q_predictions)
            summary[f'{q_name}_total'] = np.sum(q_predictions)
            
            # For higher quantiles, this represents capacity planning
            if q_value >= 0.75:
                summary[f'{q_name}_capacity_need'] = np.sum(q_predictions)
        
    else:
        # Standard model prediction
        print("Generating predictions...")
        
        # Generate predictions
        predictions = model.predict(X_data)
        
        # Create DataFrame with predictions
        results = X_data.copy()
        results[f'predicted_{target_col}'] = predictions
        
        # Calculate summary statistics
        summary = {
            'model_name': model_name,
            'target': target_col,
            'mean_prediction': np.mean(predictions),
            'median_prediction': np.median(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'total_resource_needs': np.sum(predictions)
        }
    
    print("\nResource Prediction Summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    return results, summary

def main(data_dir="data", output_dir="models", save_plots=True, use_cross_validation=False, tune_models=False):
    """Main function for training and evaluating resource utilization models"""
    
    # Define paths
    processed_dir = os.path.join(data_dir, 'processed')
    models_dir = os.path.join(data_dir, '..', 'models')
    plots_dir = os.path.join(data_dir, '..', 'reports', 'figures')
    raw_dir = os.path.join(data_dir, 'raw')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    #Test
    # Print raw_dir
    print("raw dir is: ", raw_dir)
    print("processed dir is: ", processed_dir)

    # Load data
    X_train, X_test, y_train, y_test, feature_names, target_col = load_data(processed_dir)
    
    feature_names = X_train.columns.tolist()

    # Define models to train
    models = {}
    
    # First train default models
    print("\n=== Training models with default parameters ===")
    models['Linear Regression'] = train_linear_regression(X_train, y_train)
    models['Ridge Regression'] = train_ridge_regression(X_train, y_train)
    models['Elastic Net'] = train_elastic_net(X_train, y_train)
    models['Random Forest'] = train_random_forest_regressor(X_train, y_train)
    models['LightGBM'] = train_lightgbm_regressor(X_train, y_train)
    
    # Train neural network regressor if dataset is not too large
    if X_train.shape[0] * X_train.shape[1] < 1000000:  # Adjust threshold as needed
        models['Neural Network'] = train_mlp_regressor(X_train, y_train)
    else:
        print("Dataset too large for Neural Network, skipping...")
    
    # Train quantile regression models
    models['Quantile Regression'] = train_quantile_regressor(X_train, y_train)
    
    # Perform hyperparameter tuning if requested
    if tune_models:
        print("\n=== Performing hyperparameter tuning ===")
        print("This may take a while...")
        
        # Create sample for tuning if data is large
        if X_train.shape[0] > 10000:  # Arbitrary threshold, adjust as needed
            from sklearn.model_selection import train_test_split
            print("Using a subset of data for hyperparameter tuning...")
            X_tune, _, y_tune, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)
            print(f"Tuning subset size: {X_tune.shape[0]} samples")
        else:
            X_tune, y_tune = X_train, y_train
        
        # Tune ridge regression
        print("\nTuning Ridge Regression...")
        models['Ridge Regression (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='ridge')
        
        # Tune elastic net
        print("\nTuning Elastic Net...")
        models['Elastic Net (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='elastic_net')
        
        # Tune random forest
        print("\nTuning Random Forest...")
        models['Random Forest (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='random_forest')
        
        # Tune LightGBM
        print("\nTuning LightGBM...")
        models['LightGBM (Tuned)'] = tune_hyperparameters(
            X_tune, y_tune, model_type='lightgbm')
        
        # Tune Neural Network if it was trained
        if 'Neural Network' in models:
            print("\nTuning Neural Network...")
            models['Neural Network (Tuned)'] = tune_hyperparameters(
                X_tune, y_tune, model_type='mlp')
    
    # Add cross-validation if requested
    if use_cross_validation:
        print("\n=== Running cross-validation ===")
        cv_results = {}
        
        # Define models for cross-validation
        cv_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Elastic Net': ElasticNet(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        # Clean feature names for LightGBM
        X_train_clean = clean_feature_names(X_train)
        
        # Run cross-validation for each model
        for name, model in cv_models.items():
            print(f"\nRunning 5-fold cross-validation for {name}...")
            
            # Use clean feature names for LightGBM
            if name == 'LightGBM':
                X_cv = X_train_clean
            else:
                X_cv = X_train
                
            # Calculate cross-validation scores
            cv_scores = cross_val_score(
                model, X_cv, y_train, cv=5, scoring='r2'
            )
            
            cv_results[name] = {
                'mean_r2': cv_scores.mean(),
                'std_r2': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            print(f"  Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Save cross-validation results
        cv_results_df = pd.DataFrame([{
            'model': name,
            'mean_r2': results['mean_r2'],
            'std_r2': results['std_r2']
        } for name, results in cv_results.items()])
        
        cv_results_df.to_csv(os.path.join(plots_dir, f'{target_col}_cv_results.csv'), index=False)
        
        print("\nCross-validation results:")
        print(cv_results_df.sort_values('mean_r2', ascending=False))
    
    # Evaluate models
    metrics_list = []
    for name, model in models.items():
        # Handle quantile model differently
        if name == 'Quantile Regression':
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test, name)
            metrics_list.append(metrics)
            
            # Plot quantile predictions
            quant_plot = plot_quantile_predictions(y_test, model, X_test, name)
            quant_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_quantiles.png'))
            plt.close()
            
            # Use median (q50) model for residuals plot if available
            if 'q50' in model:
                y_pred = model['q50'].predict(X_test)
                # Plot residuals
                residuals_plot = plot_residuals(y_test, y_pred, f"{name} (q50)")
                residuals_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_residuals.png'))
                plt.close()
            
            # Plot feature importance if possible
            try:
                fi_plot, fi_df = plot_feature_importance(model, feature_names, name)
                if fi_plot is not None:
                    fi_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_feature_importance.png'))
                    plt.close()
                    
                    # Save feature importance to CSV
                    fi_df.to_csv(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_feature_importance.csv'), index=False)
            except Exception as e:
                print(f"Error plotting feature importance for {name}: {e}")
            
            # Save model
            save_model(model, os.path.join(models_dir, f'{name.replace(" ", "_").lower()}_{target_col}_model.pkl'))
            
        else:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test, name)
            metrics_list.append(metrics)
            
            # Plot actual vs predicted
            actual_vs_pred_plot = plot_actual_vs_predicted(y_test, y_pred, name)
            actual_vs_pred_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_actual_vs_pred.png'))
            plt.close()
            
            # Plot residuals
            residuals_plot = plot_residuals(y_test, y_pred, name)
            residuals_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_residuals.png'))
            plt.close()
            
            # Plot feature importance if available
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                try:
                    fi_plot, fi_df = plot_feature_importance(model, feature_names, name)
                    if fi_plot is not None:
                        fi_plot.savefig(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_feature_importance.png'))
                        plt.close()
                        
                        # Save feature importance to CSV
                        fi_df.to_csv(os.path.join(plots_dir, f'{name.replace(" ", "_").lower()}_{target_col}_feature_importance.csv'), index=False)
                except Exception as e:
                    print(f"Error plotting feature importance for {name}: {e}")
            
            # Save model
            save_model(model, os.path.join(models_dir, f'{name.replace(" ", "_").lower()}_{target_col}_model.pkl'))
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(plots_dir, f'{target_col}_model_comparison.csv'), index=False)
    
    # Print model comparison
    print("\nModel Comparison:")
    comparison_cols = ['model_name', 'r2', 'rmse', 'mae', 'mape']
    print(metrics_df[comparison_cols].sort_values('r2', ascending=False))
    
    # Find best model based on R²
    best_model_idx = metrics_df['r2'].argmax()
    best_model_name = metrics_df.iloc[best_model_idx]['model_name']
    print(f"\nBest model based on R²: {best_model_name}")
    
    # Generate resource predictions using best model
    best_model = models[best_model_name]
    resource_predictions, resource_summary = predict_resource_needs(
        best_model, X_test, best_model_name, target_col
    )
    
    # Save resource predictions
    resource_predictions.to_csv(os.path.join(plots_dir, f'{target_col}_predictions.csv'), index=False)
    
    # Save resource summary
    pd.DataFrame([resource_summary]).to_csv(os.path.join(plots_dir, f'{target_col}_summary.csv'), index=False)
    
    # Also generate predictions using quantile regression for capacity planning
    if 'Quantile Regression' in models:
        print("\nGenerating capacity planning predictions using Quantile Regression...")
        quantile_predictions, quantile_summary = predict_resource_needs(
            models['Quantile Regression'], X_test, 'Quantile Regression', target_col
        )
        
        # Save capacity planning predictions
        quantile_predictions.to_csv(os.path.join(plots_dir, f'{target_col}_quantile_predictions.csv'), index=False)
        
        # Save capacity planning summary
        pd.DataFrame([quantile_summary]).to_csv(os.path.join(plots_dir, f'{target_col}_quantile_summary.csv'), index=False)
    
    print("\nResource utilization model evaluation complete!")
    return metrics_df, resource_summary

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train resource utilization prediction models')
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='Directory containing processed data')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--cv', action='store_true',
                        help='Perform cross-validation')
    args = parser.parse_args()
    
    # Call main function
    main(args.data_dir, tune_models=args.tune, use_cross_validation=args.cv)