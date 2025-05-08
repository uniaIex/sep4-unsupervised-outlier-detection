That's an excellent question! Yes, you can save the trained Isolation Forest model and use it later to evaluate new data points one by one. This is very useful for real-time applications where you need to classify new procurement entries as they come in.

Let me add this functionality to the script:

I've added the functionality you requested to save the trained Isolation Forest model and use it to evaluate new data points one by one. This is extremely useful for real-time analysis of procurement data as it comes in.

## New Features Added

### 1. Model Saving and Loading

I've implemented two functions:
- `save_model()`: Saves the trained model along with necessary metadata (feature columns, types, etc.)
- `load_model()`: Loads a previously saved model for use with new data

### 2. Single Record Prediction

The `predict_single_record()` function allows you to pass just one row of procurement data and get an instant prediction:
- Takes a single data record as input (dict, pandas.Series, or single-row DataFrame)
- Returns a boolean indicating if it's an outlier and an anomaly score
- Handles missing features and data type conversion automatically

### 3. Usage Example in Main Script

I've updated the main script to:
- Save the model after training
- Demonstrate how to load the model
- Show how to use it with a sample record

### 4. Standalone Usage Script

I've also created a separate script `procurement_outlier_detection_usage.py` that provides:
- Command-line interface for processing new data
- Ability to process entire CSV files in batch mode
- Option to process a single record through command line arguments

## How to Use This for Production

### Training and Saving the Model

1. First, run the main script to train the model on your historical procurement data:
   ```
   python procurement_outlier_detection.py
   ```

2. This will analyze your data, detect outliers, and save the trained model to `isolation_forest_model.pkl`.

### Using the Model with New Data

#### Option 1: Within your own code

```python
from procurement_outlier_detection_usage import load_model, predict_single_record

# Load the saved model
model_package = load_model('isolation_forest_model.pkl')

# Create a new procurement record
new_record = {
    'VALUE_EURO': 5000000,
    'CPV': '45000000',
    'CRIT_PRICE_WEIGHT': 60,
    # ... other fields from your dataset
}

# Check if it's an outlier
is_outlier, score = predict_single_record(new_record, model_package)

if is_outlier:
    print("ALERT: Potential procurement irregularity detected!")
    print(f"Anomaly score: {score}")
else:
    print("Record appears normal")
```

#### Option 2: Using the command-line utility

Process a CSV file containing new records:
```
python procurement_outlier_detection_usage.py --csv new_procurements.csv results.csv
```

Process a single record directly from command line:
```
python procurement_outlier_detection_usage.py --record VALUE_EURO=5000000 CPV=45000000 CRIT_PRICE_WEIGHT=60
```

This implementation makes it easy to integrate the outlier detection capability into any system - whether you're analyzing batches of new data or evaluating individual procurement records in real-time.

Would you like me to explain any specific part of the implementation in more detail?