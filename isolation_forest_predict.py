import pandas as pd
import numpy as np
import pickle
import sys
import csv
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('isolation_forest_prediction.log')
    ]
)

def load_model(filename='isolation_forest_model.pkl'):
    """
    Load the trained model along with feature information
    """
    try:
        with open(filename, 'rb') as f:
            model_package = pickle.load(f)
        
        logging.info(f"Model loaded successfully from '{filename}'")
        return model_package
    except FileNotFoundError:
        logging.error(f"Model file '{filename}' not found")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def safely_convert_to_numeric(df, numerical_columns):
    """
    Safely convert columns to numeric types
    """
    for col in numerical_columns:
        if col in df.columns:
            # Store original column
            original = df[col].copy()
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Log if many values were converted to NaN
                na_count = df[col].isna().sum()
                if na_count > 0:
                    logging.warning(f"Column '{col}': {na_count} values couldn't be converted to numeric and were set to NaN")
            except Exception as e:
                # Restore original column if conversion completely failed
                df[col] = original
                logging.error(f"Failed to convert column '{col}' to numeric: {str(e)}")
    return df

def predict_batch(df, model_package, chunk_size=1000):
    """
    Make predictions for a batch of records
    """
    # Extract model components
    model = model_package['model']
    feature_columns = model_package['feature_columns']
    numerical_features = model_package['numerical_features']
    categorical_features = model_package['categorical_features']
    
    # Keep track of missing columns
    existing_columns = set(df.columns)
    missing_columns = set(feature_columns) - existing_columns
    
    if missing_columns:
        logging.warning(f"Missing {len(missing_columns)} feature columns in dataset: {', '.join(list(missing_columns)[:5])}...")
        # Add missing columns with NaN values
        for col in missing_columns:
            df[col] = np.nan
    
    # Select only needed columns and handle any that might be missing
    features_df = pd.DataFrame()
    for col in feature_columns:
        if col in df.columns:
            features_df[col] = df[col]
        else:
            features_df[col] = np.nan
            
    # Convert numerical features
    features_df = safely_convert_to_numeric(features_df, numerical_features)
    
    # Handle missing values - fill with median for numeric, most frequent for categorical
    for col in numerical_features:
        if col in features_df.columns and features_df[col].isna().any():
            median_val = features_df[col].median()
            if np.isnan(median_val):  # If median is also NaN
                features_df[col] = features_df[col].fillna(0)
            else:
                features_df[col] = features_df[col].fillna(median_val)
    
    for col in categorical_features:
        if col in features_df.columns and features_df[col].isna().any():
            features_df[col] = features_df[col].fillna(features_df[col].mode().iloc[0] if not features_df[col].mode().empty else "UNKNOWN")
    
    # Process in chunks to avoid memory issues with large datasets
    total_records = len(features_df)
    all_predictions = []
    all_scores = []
    
    try:
        for i in range(0, total_records, chunk_size):
            end_idx = min(i + chunk_size, total_records)
            chunk = features_df.iloc[i:end_idx]
            
            logging.info(f"Processing records {i+1} to {end_idx} of {total_records}...")
            
            # Make predictions
            try:
                predictions = model.predict(chunk)
                all_predictions.extend(predictions)
                
                # Get anomaly scores if possible
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(chunk)
                    all_scores.extend(scores)
                elif hasattr(model, 'score_samples'):
                    scores = model.score_samples(chunk)
                    all_scores.extend(scores)
                else:
                    all_scores.extend([None] * len(predictions))
            except Exception as e:
                logging.error(f"Error during prediction for chunk {i}-{end_idx}: {str(e)}")
                # Fill with default values for this chunk
                all_predictions.extend([1] * (end_idx - i))  # Default to normal
                all_scores.extend([None] * (end_idx - i))
                
        # Convert predictions to outlier flags (1: normal, -1: outlier)
        is_outlier = [pred == -1 for pred in all_predictions]
        
        return is_outlier, all_scores
    except Exception as e:
        logging.error(f"Error in batch prediction: {str(e)}")
        return [False] * total_records, [None] * total_records

def predict_from_csv(csv_file, model_file='isolation_forest_model.pkl', output_file=None, chunk_size=10000):
    """
    Process a CSV file and predict outliers for each record
    """
    # Load the model
    model_package = load_model(model_file)
    if not model_package:
        return None
    
    try:
        # First, try to determine the file size to decide on loading strategy
        file_size = os.path.getsize(csv_file) / (1024 * 1024)  # Size in MB
        logging.info(f"CSV file size: {file_size:.2f} MB")
        
        large_file = file_size > 500  # Consider files > 500MB as large
        
        if large_file:
            logging.info("Large file detected - using chunked processing")
            return process_large_csv(csv_file, model_package, output_file, chunk_size)
        else:
            # For smaller files, load the entire CSV
            try:
                # First try with standard reading
                df = pd.read_csv(csv_file)
            except Exception as e:
                logging.warning(f"Standard CSV reading failed: {str(e)}. Trying with more flexible parameters...")
                try:
                    # Try with more flexible parameters
                    df = pd.read_csv(csv_file, 
                                   on_bad_lines='warn',
                                   quoting=csv.QUOTE_NONE, 
                                   escapechar='\\', 
                                   low_memory=False,
                                   encoding='utf-8')
                except Exception as e2:
                    logging.error(f"Failed to load CSV with flexible parameters: {str(e2)}")
                    try:
                        # Last attempt with minimal assumptions
                        df = pd.read_csv(csv_file, 
                                       encoding='utf-8', 
                                       sep=None, 
                                       engine='python',
                                       dtype=str)
                    except Exception as e3:
                        logging.error(f"All attempts to load CSV failed: {str(e3)}")
                        return None
            
            logging.info(f"Loaded {len(df)} records from {csv_file}")
            
            # Process the entire dataframe
            is_outliers, scores = predict_batch(df, model_package)
            
            # Add results to dataframe
            df['is_outlier'] = is_outliers
            df['anomaly_score'] = scores
            
            # Save results if output file specified
            if output_file:
                try:
                    df.to_csv(output_file, index=False)
                    logging.info(f"Saved results to {output_file}")
                except Exception as e:
                    logging.error(f"Failed to save results: {str(e)}")
                    try:
                        # Try saving without index in case of problematic columns
                        output_file_backup = output_file.replace('.csv', '_backup.csv')
                        df.to_csv(output_file_backup, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        logging.info(f"Saved backup results to {output_file_backup}")
                    except:
                        logging.error("All attempts to save results failed")
            
            # Print summary
            outlier_count = sum(is_outliers)
            logging.info(f"Outlier detection complete:")
            logging.info(f"  Total records: {len(df)}")
            logging.info(f"  Outliers detected: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
            
            return df
            
    except Exception as e:
        logging.error(f"Error processing CSV file: {str(e)}")
        return None

def process_large_csv(csv_file, model_package, output_file, chunk_size):
    """
    Process a large CSV file in chunks
    """
    logging.info(f"Processing large CSV file in chunks of {chunk_size} rows")
    
    # Initialize counters
    total_processed = 0
    total_outliers = 0
    
    # Create an output file
    if output_file:
        output_mode = 'w'  # Write mode for first chunk
        header = True      # Write header only for first chunk
    
    # Process file in chunks
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, 
                                          on_bad_lines='warn', low_memory=False)):
            start_idx = chunk_num * chunk_size
            end_idx = start_idx + len(chunk)
            
            logging.info(f"Processing chunk {chunk_num+1}: rows {start_idx}-{end_idx}")
            
            # Process chunk
            is_outliers, scores = predict_batch(chunk, model_package)
            
            # Add results
            chunk['is_outlier'] = is_outliers
            chunk['anomaly_score'] = scores
            
            # Count outliers
            chunk_outliers = sum(is_outliers)
            total_outliers += chunk_outliers
            total_processed += len(chunk)
            
            logging.info(f"  Chunk {chunk_num+1} complete: {chunk_outliers} outliers in {len(chunk)} records")
            
            # Append results to output file
            if output_file:
                try:
                    chunk.to_csv(output_file, mode=output_mode, index=False, header=header)
                    # Next chunks should append without header
                    output_mode = 'a'
                    header = False
                except Exception as e:
                    logging.error(f"Failed to save chunk {chunk_num+1}: {str(e)}")
        
        # Print final summary
        if total_processed > 0:
            logging.info(f"Large file processing complete:")
            logging.info(f"  Total records processed: {total_processed}")
            logging.info(f"  Total outliers detected: {total_outliers} ({total_outliers/total_processed*100:.2f}%)")
        else:
            logging.warning("No records were processed")
            
        return True
            
    except Exception as e:
        logging.error(f"Error during large file processing: {str(e)}")
        return False

def predict_single_from_dict(data_dict, model_file='isolation_forest_model.pkl'):
    """
    Process a single record provided as a dictionary
    """
    # Load the model
    model_package = load_model(model_file)
    if not model_package:
        return None, None
    
    try:
        # Convert to DataFrame
        record_df = pd.DataFrame([data_dict])
        
        # Extract model components
        feature_columns = model_package['feature_columns']
        numerical_features = model_package['numerical_features']
        
        # Check for missing features
        missing_features = [col for col in feature_columns if col not in record_df.columns]
        if missing_features:
            logging.warning(f"Missing features in input record: {missing_features}")
            for col in missing_features:
                record_df[col] = np.nan
        
        # Convert numeric features
        record_df = safely_convert_to_numeric(record_df, numerical_features)
        
        # Make prediction
        is_outliers, scores = predict_batch(record_df, model_package)
        
        is_outlier = is_outliers[0] if is_outliers else None
        score = scores[0] if scores else None
        
        if is_outlier is not None:
            logging.info(f"Prediction result: {'OUTLIER' if is_outlier else 'NORMAL'}")
            if score is not None:
                logging.info(f"Anomaly score: {score:.4f} (more negative = more anomalous)")
        else:
            logging.error("Prediction failed")
        
        return is_outlier, score
        
    except Exception as e:
        logging.error(f"Error predicting single record: {str(e)}")
        return None, None

def main():
    # Set up argument handling
    if len(sys.argv) < 2:
        print("Usage:")
        print("  1. Process a CSV file: python script.py --csv input.csv [output.csv] [model.pkl]")
        print("  2. Process a single record: python script.py --record KEY1=VALUE1 KEY2=VALUE2 ... [model.pkl]")
        return
    
    # Handle CSV file processing
    if sys.argv[1] == '--csv':
        if len(sys.argv) < 3:
            print("Error: CSV file path required")
            return
        
        input_csv = sys.argv[2]
        output_csv = sys.argv[3] if len(sys.argv) > 3 else None
        model_file = sys.argv[4] if len(sys.argv) > 4 else 'isolation_forest_model.pkl'
        
        logging.info(f"Processing CSV file: {input_csv}")
        logging.info(f"Output file: {output_csv if output_csv else 'None (results will not be saved)'}")
        logging.info(f"Model file: {model_file}")
        
        result = predict_from_csv(input_csv, model_file=model_file, output_file=output_csv)
        if result is not None:
            logging.info("CSV processing completed successfully")
        else:
            logging.error("CSV processing failed")
    
    # Handle single record processing
    elif sys.argv[1] == '--record':
        if len(sys.argv) < 3:
            print("Error: Record fields required (KEY1=VALUE1 KEY2=VALUE2 ...)")
            return
        
        # Find model file if provided
        model_args = [arg for arg in sys.argv if arg.endswith('.pkl')]
        if model_args:
            model_file = model_args[0]
            sys.argv.remove(model_file)
        else:
            model_file = 'isolation_forest_model.pkl'
        
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
        
        logging.info(f"Processing single record with {len(record)} fields")
        predict_single_from_dict(record, model_file=model_file)
    
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Use --csv or --record")

if __name__ == "__main__":
    main()