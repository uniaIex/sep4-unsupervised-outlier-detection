import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """
    Load the procurement dataset and preprocess it for outlier detection
    """
    print("Loading dataset...")
    # Load the dataset with more robust error handling
    try:
        # Try with error_bad_lines=False (renamed to on_bad_lines in newer pandas)
        try:
            # For newer pandas versions
            df = pd.read_csv(file_path, on_bad_lines='warn', quoting=csv.QUOTE_NONE, 
                             escapechar='\\', engine='python', encoding='utf-8')
        except TypeError:
            # For older pandas versions
            df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True, 
                             quoting=csv.QUOTE_NONE, escapechar='\\', engine='python', encoding='utf-8')
            
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        print("Note: Some rows may have been skipped due to parsing issues.")
    except Exception as e:
        print(f"Error loading dataset with standard approach: {e}")
        print("Trying alternative parsing approach...")
        
        try:
            # Try a more lenient approach with the C engine
            df = pd.read_csv(file_path, sep=',', delimiter=None, header=0, 
                             engine='c', encoding='utf-8', dtype=str,
                             quoting=csv.QUOTE_MINIMAL, na_values=[''])
            print(f"Dataset loaded with alternative method: {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e2:
            print(f"Error loading dataset with alternative approach: {e2}")
            
            # Last resort: try to read with the most flexible settings
            try:
                df = pd.read_csv(file_path, sep=None, delimiter=None, header=0,
                                engine='python', encoding='utf-8', dtype=str,
                                quoting=csv.QUOTE_NONE)
                print(f"Dataset loaded with last-resort method: {df.shape[0]} rows and {df.shape[1]} columns.")
            except Exception as e3:
                print(f"All attempts to load the dataset failed: {e3}")
                return None
    
    # Display basic information
    print("\nDataset information:")
    print(f"Number of records: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Select relevant numerical and categorical features for outlier detection
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    # Remove ID columns and timestamp columns from features
    exclude_patterns = ['ID_', 'DT_', 'URL', 'CODE']
    numerical_features = [col for col in numerical_features if not any(pattern in col for pattern in exclude_patterns)]
    
    # Keep only categorical features with reasonable cardinality
    categorical_features_filtered = []
    for col in categorical_features:
        if df[col].nunique() < 50 and not any(pattern in col for pattern in exclude_patterns):
            categorical_features_filtered.append(col)
    
    print(f"\nSelected {len(numerical_features)} numerical features and {len(categorical_features_filtered)} categorical features for analysis.")
    
    # Convert data types for numerical features
    print("\nConverting data types...")
    for col in numerical_features:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Error converting column {col} to numeric: {e}")
            # If conversion fails, remove from numerical features
            numerical_features.remove(col)
    
    # Create a feature set for outlier detection
    try:
        feature_df = df[numerical_features + categorical_features_filtered].copy()
    except KeyError as e:
        print(f"Error creating feature DataFrame: {e}")
        print("Available columns:", df.columns.tolist())
        # Try to recover by using only available columns
        available_features = [col for col in numerical_features + categorical_features_filtered if col in df.columns]
        if not available_features:
            print("No usable features found. Cannot proceed with analysis.")
            return None
        feature_df = df[available_features].copy()
        # Update feature lists
        numerical_features = [col for col in numerical_features if col in available_features]
        categorical_features_filtered = [col for col in categorical_features_filtered if col in available_features]
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in feature_df.columns:
        missing_pct = feature_df[col].isnull().mean() * 100
        if missing_pct > 0:
            print(f"Column {col}: {missing_pct:.2f}% missing values")
    
    return df, feature_df, numerical_features, categorical_features_filtered

def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Build a preprocessing pipeline for the dataset
    """
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def detect_outliers(feature_df, numerical_features, categorical_features):
    """
    Detect outliers using Isolation Forest
    """
    print("\nBuilding preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Create and fit the isolation forest model
    print("Fitting Isolation Forest model...")
    
    try:
        # Use a simple approach if there are issues with the pipeline
        if len(feature_df) < 10 or (not numerical_features and not categorical_features):
            print("Warning: Limited data or features. Using simplified approach.")
            # Use only numeric columns that can be converted safely
            numeric_df = feature_df.select_dtypes(include=['number']).copy()
            
            if numeric_df.empty:
                print("No numeric features available. Attempting to convert some columns...")
                for col in feature_df.columns[:5]:  # Try first 5 columns
                    try:
                        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                        if not feature_df[col].isna().all():
                            numeric_df[col] = feature_df[col]
                    except:
                        pass
            
            if numeric_df.empty:
                print("Could not create any numeric features. Cannot proceed with outlier detection.")
                dummy_outliers = np.zeros(len(feature_df), dtype=bool)
                return None, dummy_outliers
            
            # Fill missing values with median
            numeric_df = numeric_df.fillna(numeric_df.median())
            
            # Apply isolation forest directly
            clf = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.05,
                random_state=42
            )
            outlier_predictions = clf.fit_predict(numeric_df)
            isolation_forest = clf
            
        else:
            # Standard approach with full pipeline
            isolation_forest = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', IsolationForest(
                    n_estimators=100,
                    max_samples='auto',
                    contamination=0.05,  # Expecting about 5% outliers
                    random_state=42
                ))
            ])
            
            # Fit the model and predict outliers
            outlier_predictions = isolation_forest.fit_predict(feature_df)
    
    except Exception as e:
        print(f"Error in outlier detection: {e}")
        print("Falling back to basic approach...")
        
        # Very basic fallback: use only simple numeric features
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            print("No numeric columns available for fallback approach.")
            dummy_outliers = np.zeros(len(feature_df), dtype=bool)
            return None, dummy_outliers
        
        # Use a subset of numeric columns
        simple_df = feature_df[numeric_cols].fillna(0)
        
        # Simple isolation forest
        clf = IsolationForest(contamination=0.05, random_state=42)
        outlier_predictions = clf.fit_predict(simple_df)
        isolation_forest = clf
    
    # Convert predictions to binary labels (1: inlier, -1: outlier)
    outliers = outlier_predictions == -1
    
    print(f"Detected {sum(outliers)} outliers out of {len(feature_df)} records ({sum(outliers)/len(feature_df)*100:.2f}%)")
    
    return isolation_forest, outliers

def analyze_outliers(df, feature_df, outliers, numerical_features, categorical_features):
    """
    Analyze and visualize the detected outliers
    """
    # Add outlier flag to the original dataframe
    df_with_outliers = df.copy()
    df_with_outliers['is_outlier'] = outliers
    
    try:
        # Analyze numerical features
        print("\nAnalyzing numerical features for outliers vs. non-outliers:")
        for col in numerical_features:
            if col in feature_df.columns:
                try:
                    outlier_values = df_with_outliers[df_with_outliers['is_outlier']][col]
                    non_outlier_values = df_with_outliers[~df_with_outliers['is_outlier']][col]
                    
                    if not outlier_values.empty and not non_outlier_values.empty:
                        # Check if values can be treated as numeric
                        if pd.api.types.is_numeric_dtype(outlier_values) and pd.api.types.is_numeric_dtype(non_outlier_values):
                            print(f"\nFeature: {col}")
                            
                            # Safely calculate statistics
                            try:
                                o_mean = outlier_values.mean()
                                o_std = outlier_values.std()
                                o_min = outlier_values.min()
                                o_max = outlier_values.max()
                                
                                no_mean = non_outlier_values.mean()
                                no_std = non_outlier_values.std()
                                no_min = non_outlier_values.min()
                                no_max = non_outlier_values.max()
                                
                                print(f"  Outliers: mean={o_mean:.2f}, std={o_std:.2f}, min={o_min:.2f}, max={o_max:.2f}")
                                print(f"  Non-outliers: mean={no_mean:.2f}, std={no_std:.2f}, min={no_min:.2f}, max={no_max:.2f}")
                            except Exception as e:
                                print(f"  Error calculating statistics for {col}: {e}")
                except Exception as e:
                    print(f"  Error analyzing feature {col}: {e}")
        
        # Analyze categorical features
        print("\nAnalyzing categorical features for outliers vs. non-outliers:")
        for col in categorical_features:
            if col in feature_df.columns:
                try:
                    # Calculate the distribution difference
                    outlier_dist = df_with_outliers[df_with_outliers['is_outlier']][col].value_counts(normalize=True)
                    non_outlier_dist = df_with_outliers[~df_with_outliers['is_outlier']][col].value_counts(normalize=True)
                    
                    # Find categories with significant differences
                    all_categories = set(outlier_dist.index) | set(non_outlier_dist.index)
                    significant_diffs = []
                    
                    for category in all_categories:
                        try:
                            outlier_pct = outlier_dist.get(category, 0) * 100
                            non_outlier_pct = non_outlier_dist.get(category, 0) * 100
                            diff = abs(outlier_pct - non_outlier_pct)
                            if diff > 10:  # More than 10% difference
                                significant_diffs.append((category, outlier_pct, non_outlier_pct, diff))
                        except:
                            continue
                    
                    if significant_diffs:
                        print(f"\nFeature: {col}")
                        for category, outlier_pct, non_outlier_pct, diff in sorted(significant_diffs, key=lambda x: x[3], reverse=True):
                            print(f"  Category '{category}': outliers={outlier_pct:.1f}%, non-outliers={non_outlier_pct:.1f}%, diff={diff:.1f}%")
                except Exception as e:
                    print(f"  Error analyzing categorical feature {col}: {e}")
    except Exception as e:
        print(f"Error in outlier analysis: {e}")
    
    return df_with_outliers

def visualize_outliers(df_with_outliers, numerical_features):
    """
    Create visualizations to better understand the outliers
    """
    print("\nGenerating visualizations...")
    
    try:
        # Ensure we have valid numerical features
        valid_features = []
        for feature in numerical_features:
            if feature in df_with_outliers.columns:
                try:
                    # Check if the feature can be treated as numeric
                    df_with_outliers[feature] = pd.to_numeric(df_with_outliers[feature], errors='coerce')
                    # Check if we have enough non-null values
                    if df_with_outliers[feature].notna().sum() > 10:
                        valid_features.append(feature)
                except:
                    pass
        
        if not valid_features:
            print("No valid numerical features for visualization. Skipping visualizations.")
            return
        
        # Select most important numerical features (up to 5)
        vis_features = valid_features[:min(5, len(valid_features))]
        
        # Set up the figure
        fig, axes = plt.subplots(len(vis_features), 1, figsize=(12, 4*len(vis_features)))
        if len(vis_features) == 1:
            axes = [axes]
        
        # Plot histograms for selected features
        for i, feature in enumerate(vis_features):
            try:
                sns.histplot(data=df_with_outliers, x=feature, hue='is_outlier', 
                             bins=30, ax=axes[i], multiple='layer', palette=['blue', 'red'])
                axes[i].set_title(f'Distribution of {feature} by Outlier Status')
                axes[i].legend(['Normal', 'Outlier'])
            except Exception as e:
                print(f"Error plotting histogram for {feature}: {e}")
        
        plt.tight_layout()
        try:
            plt.savefig('outlier_distributions.png')
            print("Visualization saved as 'outlier_distributions.png'")
        except Exception as e:
            print(f"Error saving histogram visualization: {e}")
        
        # Create a scatter plot matrix for the top numerical features
        if len(vis_features) >= 2:
            try:
                plt.figure(figsize=(12, 10))
                scatter_features = vis_features[:min(4, len(vis_features))]
                
                # Create a clean DataFrame for plotting
                plot_df = df_with_outliers[scatter_features + ['is_outlier']].copy()
                
                # Handle any remaining NaN values
                for col in scatter_features:
                    median_val = plot_df[col].median()
                    plot_df[col] = plot_df[col].fillna(median_val)
                
                sns.pairplot(plot_df, vars=scatter_features, hue='is_outlier', palette=['blue', 'red'])
                plt.savefig('outlier_scatter_matrix.png')
                print("Scatter plot matrix saved as 'outlier_scatter_matrix.png'")
            except Exception as e:
                print(f"Error creating scatter plot matrix: {e}")
    except Exception as e:
        print(f"Error in visualization function: {e}")

def save_model(model, feature_columns, numerical_features, categorical_features, filename='isolation_forest_model.pkl'):
    """
    Save the trained model along with feature information for later use
    """
    import pickle
    
    # Create a dictionary containing all necessary components
    model_package = {
        'model': model,
        'feature_columns': feature_columns,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"\nModel and feature information saved to '{filename}'")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filename='isolation_forest_model.pkl'):
    """
    Load the trained model along with feature information
    """
    import pickle
    
    try:
        with open(filename, 'rb') as f:
            model_package = pickle.load(f)
        
        print(f"Model loaded successfully from '{filename}'")
        return model_package
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_single_record(record, model_package):
    """
    Predict if a single record is an outlier
    
    Parameters:
    -----------
    record : dict or pd.Series
        A single data record with feature values
    model_package : dict
        Model package containing model and feature information
        
    Returns:
    --------
    is_outlier : bool
        True if the record is predicted as an outlier, False otherwise
    anomaly_score : float
        The anomaly score (-1 to 0, with more negative being more anomalous)
    """
    import pandas as pd
    
    # Extract model components
    model = model_package['model']
    feature_columns = model_package['feature_columns']
    numerical_features = model_package['numerical_features']
    categorical_features = model_package['categorical_features']
    
    # Convert record to DataFrame if it's a dict
    if isinstance(record, dict):
        record_df = pd.DataFrame([record])
    else:
        record_df = pd.DataFrame([record.to_dict()])
    
    # Check if all required features are present
    missing_features = [col for col in feature_columns if col not in record_df.columns]
    if missing_features:
        print(f"Warning: Missing features in the input record: {missing_features}")
        # Add missing columns with NaN values
        for col in missing_features:
            record_df[col] = np.nan
    
    # Select only the features used during training
    record_features = record_df[feature_columns].copy()
    
    # Convert numeric features
    for col in numerical_features:
        if col in record_features.columns:
            try:
                record_features[col] = pd.to_numeric(record_features[col], errors='coerce')
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")
    
    try:
        # Make prediction (1: normal, -1: outlier)
        prediction = model.predict(record_features)[0]
        
        # Get anomaly score if possible (more negative is more anomalous)
        try:
            if hasattr(model, 'decision_function'):
                score = model.decision_function(record_features)[0]
            elif hasattr(model, 'score_samples'):
                score = model.score_samples(record_features)[0]
            else:
                score = None
        except:
            score = None
        
        # Return result
        is_outlier = (prediction == -1)
        
        return is_outlier, score
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def main():
    """
    Main function to run the outlier detection analysis
    """
    print("Procurement Data Outlier Detection using Isolation Forest")
    print("=" * 60)
    
    # File path
    file_path = "2020_utf8_data.csv"  # Update with actual file path
    
    # Load and preprocess data
    result = load_and_preprocess_data(file_path)
    
    if result is None:
        print("Failed to load dataset. Exiting.")
        return
    
    df, feature_df, numerical_features, categorical_features = result
    
    # Detect outliers
    isolation_forest, outliers = detect_outliers(feature_df, numerical_features, categorical_features)
    
    if isolation_forest is None:
        print("Outlier detection failed. Exiting.")
        return
    
    # Analyze outliers
    df_with_outliers = analyze_outliers(df, feature_df, outliers, numerical_features, categorical_features)
    
    # Visualize outliers
    visualize_outliers(df_with_outliers, numerical_features)
    
    # Export outliers
    #export_outliers(df_with_outliers)
    
    # Save the model for future use with new data
    save_model(isolation_forest, feature_df.columns.tolist(), 
               numerical_features, categorical_features)
    
    print("\nOutlier detection analysis completed!")
    
    # Example of how to use the saved model with a new record
    print("\n=== Example: Using the model with a new record ===")
    print("Loading the saved model...")
    model_package = load_model()
    
    if model_package:
        # Example: Create a sample new record (replace with actual data)
        if len(df) > 0:
            # Use first record from dataset as an example
            sample_record = df.iloc[0].copy()
            
            print("\nSample record for prediction:")
            print(f"Record ID: {sample_record.get('ID_NOTICE_CN', 'N/A')}")
            
            # Predict if it's an outlier
            is_outlier, score = predict_single_record(sample_record, model_package)
            
            if is_outlier is not None:
                print(f"Prediction result: {'OUTLIER' if is_outlier else 'NORMAL'}")
                if score is not None:
                    print(f"Anomaly score: {score:.4f} (more negative = more anomalous)")
            else:
                print("Prediction failed.")

if __name__ == "__main__":
    main()