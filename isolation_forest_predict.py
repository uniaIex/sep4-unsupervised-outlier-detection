import pandas as pd
import numpy as np
import pickle
import sys

def load_model(filename='isolation_forest_model.pkl'):
    """
    Load the trained model along with feature information
    """
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
    record : dict or pd.Series or pd.DataFrame
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
    # Extract model components
    model = model_package['model']
    feature_columns = model_package['feature_columns']
    numerical_features = model_package['numerical_features']
    categorical_features = model_package['categorical_features']
    
    # Convert record to DataFrame if it's a dict or Series
    if isinstance(record, dict):
        record_df = pd.DataFrame([record])
    elif isinstance(record, pd.Series):
        record_df = pd.DataFrame([record.to_dict()])
    elif isinstance(record, pd.DataFrame):
        if len(record) == 1:
            record_df = record.copy()
        else:
            print("Error: Input must be a single record (dict, Series, or DataFrame with 1 row)")
            return None, None
    else:
        print(f"Error: Unsupported record type: {type(record)}")
        return None, None
    
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

def predict_from_csv(csv_file, model_file='isolation_forest_model.pkl', output_file=None):
    """
    Process a CSV file and predict outliers for each record
    """
    # Load the model
    model_package = load_model(model_file)
    if not model_package:
        return
    
    # Load the CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Process each record
    results = []
    for i, row in df.iterrows():
        print(f"Processing record {i+1}/{len(df)}...", end='\r')
        is_outlier, score = predict_single_record(row, model_package)
        results.append({
            'record_index': i,
            'is_outlier': is_outlier,
            'anomaly_score': score
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original data
    output_df = pd.concat([df, results_df[['is_outlier', 'anomaly_score']]], axis=1)
    
    # Save results if output file specified
    if output_file:
        output_df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
    
    # Print summary
    outlier_count = results_df['is_outlier'].sum()
    print(f"\nOutlier detection complete:")
    print(f"  Total records: {len(df)}")
    print(f"  Outliers detected: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
    
    return output_df

def predict_single_from_dict(data_dict, model_file='isolation_forest_model.pkl'):
    """
    Process a single record provided as a dictionary
    """
    # Load the model
    model_package = load_model(model_file)
    if not model_package:
        return False, None
    
    # Process the record
    is_outlier, score = predict_single_record(data_dict, model_package)
    
    # Print results
    if is_outlier is not None:
        print(f"Prediction result: {'OUTLIER' if is_outlier else 'NORMAL'}")
        if score is not None:
            print(f"Anomaly score: {score:.4f} (more negative = more anomalous)")
        return is_outlier, score
    else:
        print("Prediction failed.")
        return None, None

def main():
    # Example usage
    if len(sys.argv) < 2:
        print("Usage:")
        print("  1. Process a CSV file: python script.py --csv input.csv [output.csv]")
        print("  2. Process a single record: python script.py --record KEY1=VALUE1 KEY2=VALUE2 ...")
        return
    
    # Handle CSV file processing
    if sys.argv[1] == '--csv':
        if len(sys.argv) < 3:
            print("Error: CSV file path required")
            return
        
        input_csv = sys.argv[2]
        output_csv = sys.argv[3] if len(sys.argv) > 3 else None
        predict_from_csv(input_csv, output_file=output_csv)
    
    # Handle single record processing
    elif sys.argv[1] == '--record':
        if len(sys.argv) < 3:
            print("Error: Record fields required (KEY1=VALUE1 KEY2=VALUE2 ...)")
            return
        
        # Parse record data from command line
        record = {}
        for arg in sys.argv[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                record[key] = value
            else:
                print(f"Warning: Ignoring invalid argument format: {arg}")
        
        if not record:
            print("Error: No valid key-value pairs provided")
            return
        
        print("Processing record with fields:", list(record.keys()))
        predict_single_from_dict(record)
    
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Use --csv or --record")

if __name__ == "__main__":
    main() 