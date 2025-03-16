# src/data/preprocess.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Add this at the top of make_dataset.py, extract_features.py, and preprocess.py
def ensure_correct_paths(data_dir):
    """
    Ensure that we're using the correct paths for raw, interim, and processed data
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
        
    Returns:
    --------
    base_dir, raw_dir, interim_dir, processed_dir
    """
    import os
    
    # Normalize the base path
    base_dir = os.path.abspath(data_dir)
    
     # If data_dir already ends with 'data', use it directly
    # Otherwise, assume we need to add subdirectories
    if os.path.basename(base_dir) == 'data':
        raw_dir = os.path.join(base_dir, 'raw')
        interim_dir = os.path.join(base_dir, 'interim')
        processed_dir = os.path.join(base_dir, 'processed')
    else:
        # If we were passed 'data/raw' directly
        raw_dir = base_dir
        interim_dir = os.path.join(os.path.dirname(base_dir), 'interim')
        processed_dir = os.path.join(os.path.dirname(base_dir), 'processed')
    
    # Ensure all directories exist
    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Make sure raw directory exists and has files
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    raw_files = os.listdir(raw_dir)
    if not raw_files:
        raise FileNotFoundError(f"No files found in raw data directory: {raw_dir}")
    
    print(f"Base directory: {base_dir}")
    print(f"Raw directory: {raw_dir}")
    print(f"Interim directory: {interim_dir}")
    print(f"Processed directory: {processed_dir}")
    print(f"Files in raw directory: {raw_files}")
    
    return base_dir, raw_dir, interim_dir, processed_dir

def calculate_los(df, data_dir):
    """
    Calculate the length of stay in days for each admission
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
        
    Returns:
    --------
    DataFrame
        DataFrame with length of stay in days for each admission
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    def calculate_stay_duration(df, start_col, end_col, duration_col):
        """Helper function to calculate length of stay"""
        df[start_col] = pd.to_datetime(df[start_col])
        df[end_col] = pd.to_datetime(df[end_col])
        df[duration_col] = (df[end_col] - df[start_col]).dt.total_seconds() / (24 * 3600)
        df[duration_col] = df[duration_col].clip(lower=0)
        return df
    
    # Load and process admissions data
    admissions = pd.read_csv(os.path.join(raw_dir, 'admissions.csv'))
    admissions = calculate_stay_duration(
        admissions, 
        'admittime', 
        'dischtime', 
        'total_los_days'
    )
    
    # Load and process ICU stays data
    icustays = pd.read_csv(os.path.join(raw_dir, 'icustays.csv'))
    icustays = calculate_stay_duration(
        icustays,
        'intime',
        'outtime',
        'icu_los_days'
    )
    
    # NOTE: it may be better to calculate total los using edregtime as the start if it precedes admittime

    # Merge admissions and ICU stays
    merged_data = pd.merge(admissions, icustays, on='hadm_id', how='left')

    # Calculate ICU length of stay ratio to total length of stay
    merged_data['icu_los_ratio'] = merged_data['icu_los_days'] / merged_data['total_los_days']
    
    # Fill missing values with 0
    merged_data['icu_los_ratio'] = merged_data['icu_los_ratio'].fillna(0)
    merged_data['icu_los_days'] = merged_data['icu_los_days'].fillna(0)

    # add merged data to full dataframe
    df = pd.merge(df, merged_data[['hadm_id', 'total_los_days', 'icu_los_days', 'icu_los_ratio']], on='hadm_id', how='left')

    return df 

def create_final_dataset(data_dir):
    """
    Create the final dataset by merging patient and admission features

    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
        
    Returns:
    --------
    DataFrame
        The final dataset with all features
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    # Check for required files
    required_files = [
        os.path.join(interim_dir, 'hf_admissions_with_readmissions.csv'),
        os.path.join(interim_dir, 'hf_patient_features.csv'),
        os.path.join(interim_dir, 'hf_admission_features.csv')
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: The following required files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nContents of interim directory:")
        try:
            print(os.listdir(interim_dir))
        except Exception as e:
            print(f"Error listing interim directory: {e}")
        
        raise FileNotFoundError(f"Missing required files. Please run the previous pipeline steps.")
    
    # Load data
    print("Loading intermediate files...")
    hf_admissions = pd.read_csv(required_files[0])
    patient_features = pd.read_csv(required_files[1])
    admission_features = pd.read_csv(required_files[2])

    # Check if 'is_readmission' exists in both dataframes to prevent duplication
    if 'is_readmission' in hf_admissions.columns and 'is_readmission' in admission_features.columns:
        # Compare the 'is_readmission' columns
        comparison = (hf_admissions[['hadm_id', 'is_readmission']].set_index('hadm_id') ==
                      admission_features[['hadm_id', 'is_readmission']].set_index('hadm_id'))['is_readmission']
        
        if not comparison.all():
            raise ValueError("The 'is_readmission' column differs between hf_admissions and admission_features.  "
                             "Resolve this discrepancy before proceeding.")
        
        # Remove duplicate target column
        admission_features = admission_features.drop('is_readmission', axis=1)
    
    # Merge all features
    print("Merging feature sets...")
    merged_data = pd.merge(hf_admissions, patient_features, on='subject_id', how='left')
    merged_data = pd.merge(merged_data, admission_features, on='hadm_id', how='left')
    
    print(f"Final dataset has {len(merged_data)} rows and {len(merged_data.columns)} columns")
    
    return merged_data

def handle_missing_values(df, id_cols, target_col, exclude_cols):
    """
    Handle missing values in the dataset by dropping rows with missing values
    only in critical columns and imputing other values where appropriate
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with missing values
    numerical_cols : list
        List of numerical column names
    id_cols : list
        List of ID column names to preserve
    target_col : str
        Target column name
    exclude_cols : list
        Columns that will be excluded from the final dataset
        
    Returns:
    --------
    DataFrame with handled missing values
    """
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Print initial distribution of target values
    if target_col in df_cleaned.columns:
        target_counts = df_cleaned[target_col].value_counts()
        print(f"Initial target distribution: {target_counts}")
        print(f"Initial positive rate: {target_counts.get(True, 0) / len(df_cleaned):.2%}")
    
    # NOTE: Future version may automatically determine which columns to drop from or use exclude_cols
    # Critical columns to use for dropping
    critical_cols = [
        'anchor_age', 'gender', 'marital_status', 'insurance', 'language',
        'hypertension', 'diabetes', 'kidney_disease', 'copd', 'coronary_artery_disease', 'obesity', 'anemia',
    ]
    
    # Only drop rows with missing values in critical columns
    critical_cols = critical_cols + [target_col] + id_cols
    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=critical_cols)
    dropped_rows = rows_before - len(df_cleaned)
    print(f"\nDropped {dropped_rows} rows with missing values in critical columns")
    print(f"Remaining rows: {len(df_cleaned)}")
    
    # Check remaining missing values
    missing_after = df_cleaned[critical_cols].isnull().sum().sum()
    if missing_after == 0:
        print("\nNo missing values remain in critical columns")
    else:
        print(f"\nWarning: {missing_after} missing values remain in critical columns")
    
    return df_cleaned

def normalize_numerical_features(df, numerical_cols, scaler=MinMaxScaler([0,1])):
    """
    Normalize numerical features using MinMaxScaler
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing numerical columns
    numerical_cols : list
        List of numerical column names
        
    Returns:
    --------
    df_normalized : pandas DataFrame
        DataFrame with normalized numerical columns
    """
    df_normalized = df.copy()
    
    # Filter to only include columns that actually exist in the dataframe
    existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if existing_numerical_cols:
        scaler = MinMaxScaler()
        df_normalized[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])
    
    df_normalized = truncate_numerical(df_normalized)

    return df_normalized

def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    categorical_cols : list
        List of categorical column names
        
    Returns:
    --------
    DataFrame with encoded features, encoder object
    """
    # Create a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Apply one-hot encoding to categorical columns
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first') # changed parameter from sparse to sparce_output, as sparse was depreciated
        encoded_features = encoder.fit_transform(df[categorical_cols])
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_features, 
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        
        # Drop original categorical columns
        df_encoded = df_encoded.drop(categorical_cols, axis=1)
        
        # Concatenate encoded features
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        return df_encoded, encoder
    
    return df_encoded, None

def create_binary_columns(df, exclude_cols):
    """
    Create binary indicators for multiple categorical columns
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    exclude_cols : list
        List of columns to exclude
        
    Returns:
    --------
    df : DataFrame
        DataFrame with new binary indicators
    exclude_cols : list 
        Updated exclude columns
    """
    # Define the mappings for binary columns
    binary_mappings = [
        ('language', 'ENGLISH', 'english_lang'),
        ('marital_status', 'MARRIED', 'is_married'),
        ('insurance', 'MEDICARE', 'insurance_Medicare'),
    ]
    
    for column_name, target_value, new_column_name in binary_mappings:
        if column_name in df.columns:
            df[new_column_name] = (df[column_name].str.upper() == target_value.upper()).astype(int)
            # Add original column to exclude cols if not already present
            if column_name not in exclude_cols:
                exclude_cols.append(column_name)
    
    return df, exclude_cols

def standardize_race(df):
    """
    Standardize race column by first word and applying manual overrides
    More research should be done on data and race categories to ensure accuracy

    Parameters:
    -----------
    df : DataFrame
        Input DataFrame

    Returns:
    --------
    DataFrame
        DataFrame with standardized race column
    """
    # Standardize race categories by first word
    if 'race' in df.columns:
        df_standard = df.copy()
        df_standard['race'] = df_standard['race'].fillna('Unknown')
        df_standard['race_standardized'] = df_standard['race'].apply(lambda x: x.split(' ')[0].split('/')[0].title())
        
        # Create mapping for any manual overrides
        # note: this is a few examples and may not cover all cases
        race_mapping = {
            'Hispanic': 'Latino',    # Map Hispanic to Latino
            'Portuguese': 'Latino',  # Map Portuguese to Latino
            'American': 'Native_American',  # Map American to Native_American
            'Native': 'Pacific_Islander',   # Map Native to Pacific_Islander
            'Patient': 'Unknown',    # Map Patient to Unknown
            'Unable': 'Unknown',     # Map Unable to Unknown
            # Mappings that should be reviewed:
            # 'WHITE - BRAZILIAN' is currently mapped to WHITE
            # 'PORTUGUESE' was assumed to mean Brazilian and mapped to LATINO
            # 'ASIAN - ASIAN INDIAN' is included in ASIAN
            # 'OTHER' and 'MULTIPLE' were left as-is
        }
        
        # Apply manual overrides while keeping other first-word mappings
        df['race'] = df_standard['race_standardized'].map(lambda x: race_mapping.get(x, x))

    return df

def truncate_numerical(df):
    """
    Preprocess features to handle extreme values, infinities, and NaNs
    
    Parameters:
    -----------
    df : DataFrame
        
    Returns:
    --------
    df : DataFrame
        DataFrame with truncated numerical features
    """
    # Make copies to avoid modifying the original data
    df_copy = df.copy()
    
    # Get numerical features
    numeric_features = df_copy.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"Processing {len(numeric_features)} numerical features")
    
    # Process each numerical column
    for col in numeric_features:
        # Round to 5 decimal places
        df_copy[col] = df_copy[col].round(decimals=5)
    
    return df_copy

def prepare_data_for_modeling(df, id_cols, target_col='is_readmission', test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into features and target,
    and then into training and testing sets
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    target_col : str, default='is_readmission'
        Target column name
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    # Separate features and target
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in DataFrame.")
        print("Available columns:")
        for col in df.columns:
            print(f"- {col}")
        raise ValueError(f"Target column '{target_col}' not found.")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"Positive class (readmissions) in training: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train):.1%})")
    print(f"Positive class (readmissions) in testing: {sum(y_test)}/{len(y_test)} ({sum(y_test)/len(y_test):.1%})")
    
    return X_train, X_test, y_train, y_test

def main(data_dir):
    """
    Main preprocessing pipeline
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    # Create final dataset
    final_data = create_final_dataset(data_dir)
    
    # Define column types
    id_cols = ['subject_id', 'hadm_id']
    target_col = 'is_readmission'
    
    # Define numerical columns explicitly
    numerical_cols = ['anchor_age', 'total_los_days', #'icu_los_days',
                      ]
    categorical_cols = ['gender', 'race',
                        ]
    
    # Exclude columns that won't be used in modeling
    exclude_cols = [
        'admittime', 'dischtime', 'deathtime', 'days_until_readmission',
        'admission_type', 'admission_location', 'discharge_location', 
        'edregtime', 'edouttime', 'hospital_expire_flag'
    ]

    # Handle missing values
    data_imputed = handle_missing_values(final_data, id_cols, target_col, exclude_cols)

    # Calculate length of stay
    data_imputed = calculate_los(data_imputed, data_dir)

    # Create binary indicators for language and marital status
    data_imputed, exclude_cols = create_binary_columns(data_imputed, exclude_cols)
    
    # Standardize race column
    data_imputed = standardize_race(data_imputed)

    # Normalize numerical features
    data_normalized = normalize_numerical_features(data_imputed, numerical_cols, scaler=MinMaxScaler([0,1]))
    
    # Encode categorical features
    data_processed, encoder = encode_categorical_features(data_normalized, categorical_cols)
    
    # Remove all excluded columns
    data_processed = data_processed.loc[:, ~data_processed.columns.isin(exclude_cols)]

    # Rename anchor_age to age if it exists in the dataframe
    if 'anchor_age' in data_processed.columns:
        data_processed = data_processed.rename(columns={'anchor_age': 'age'})
    
    # Remove ID columns after all processing
    # Trying only with subject_id for now
    data_processed = data_processed.drop(columns='subject_id')

    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(data_processed, id_cols, target_col)

    # NOTE: Drop icu_los_ratio for now due to potential data leakage
    data_processed = data_processed.drop(columns='icu_los_ratio')
    X_train = X_train.drop(columns='icu_los_ratio')
    X_test = X_test.drop(columns='icu_los_ratio')

    # Save full processed dataset
    data_processed.to_csv(os.path.join(processed_dir, 'hf_data_processed.csv'), index=False)
    
    # Save train/test splits
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print("Preprocessing complete. Files saved to processed folder.")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../../data'
    main(data_dir)