# SEP4 - Unsupervised Outlier Detection

An unsupervised machine learning project that implements Isolation Forest algorithm for outlier detection in procurement data. This project is designed to identify anomalous procurement records that may indicate potential fraud, irregularities, or unusual spending patterns.

## Overview

This project uses the Isolation Forest algorithm, an ensemble-based unsupervised anomaly detection method that isolates outliers by randomly selecting features and splitting data points. The algorithm is particularly effective for high-dimensional datasets where anomalies are rare and distinct from normal data points.

## Features

- **Unsupervised Learning**: No labeled data required for training
- **Scalable**: Efficient performance on large datasets
- **Procurement Focus**: Specifically designed for procurement data analysis
- **Batch Processing**: Support for processing multiple records from CSV files
- **Single Record Analysis**: Command-line interface for individual record evaluation
- **Visualization**: Generate distribution plots and scatter plots for outlier analysis
- **Model Persistence**: Save and load trained models for reuse

## Dataset

The project uses a subset of procurement data from `data.europa.eu`, specifically from the TED (Tenders Electronic Daily) database. The dataset includes features such as:

## Project Structure

```
├── data-preview-ted.csv              # Training dataset (subset from TED)
├── isolation_forest.py              # Model training script
├── isolation_forest_model.pkl       # Saved trained model
├── isolation_forest_predict.py      # Prediction script
├── procurement_outlier_detection_usage.py  # Main usage script
├── results.csv                      # Prediction results output
├── outlier_distributions.png        # Outlier distribution visualization
└── outlier_scatter_mix.png         # Scatter plot visualization
```

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
Since the virtual environment is not included in the repository, you'll need to install the required libraries manually:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

Common dependencies likely include:
- `scikit-learn` - For Isolation Forest implementation
- `pandas` - For data manipulation
- `numpy` - For numerical operations
- `matplotlib` - For visualization
- `seaborn` - For enhanced plotting

## Usage

### 1. Train the Model

First, train the Isolation Forest model using the provided dataset:

```bash
python isolation_forest.py
```

This script will:
- Load and preprocess the training data from `data-preview-ted.csv`
- Train the Isolation Forest model
- Save the trained model to `isolation_forest_model.pkl`

### 2. Detect Outliers

#### Process a CSV file with multiple records:
```bash
python procurement_outlier_detection_usage.py --csv new_procurements.csv results.csv
```

#### Analyze a single record from command line:
```bash
python procurement_outlier_detection_usage.py --record VALUE_EURO=5000000 CPV=45000000 CRIT_PRICE_WEIGHT=60
```

### 3. Visualize Results

The project generates visualization files to help understand the outlier patterns:
- `outlier_distributions.png` - Shows distribution of outliers across different features
- `outlier_scatter_mix.png` - Scatter plot visualization of outliers vs normal data points

## How It Works

### Isolation Forest Algorithm

The Isolation Forest algorithm works on the principle that:
1. **Anomalies are rare**: They constitute a small percentage of the data
2. **Anomalies are different**: They have feature values that differ significantly from normal observations

The algorithm:
1. Randomly selects features and split values
2. Creates isolation trees by recursively partitioning the data
3. Measures the path length required to isolate each point
4. Points that require shorter paths are more likely to be outliers

### Model Training Process

1. **Data Loading**: Reads procurement data from CSV
2. **Feature Engineering**: Processes and transforms features for optimal model performance
3. **Model Training**: Fits the Isolation Forest on the training data
4. **Model Serialization**: Saves the trained model for future use

### Prediction Process

1. **Model Loading**: Loads the pre-trained model
2. **Data Preprocessing**: Applies the same transformations used during training
3. **Outlier Detection**: Predicts whether each record is an outlier
4. **Results Export**: Saves predictions to CSV and displays console output

## Output Format

### Console Output
The prediction results are displayed in the console showing:
- Record details
- Outlier probability/score
- Classification (Normal/Outlier)

### CSV Output (`results.csv`)
Contains detailed results with columns:
- Original features
- Outlier score
- Prediction label
- Additional metadata

## Model Performance

The Isolation Forest model is particularly effective for:
- **High-dimensional data** with multiple features
- **Imbalanced datasets** where outliers are rare
- **Unsupervised scenarios** where labeled anomaly data is unavailable

## Use Cases

This outlier detection system can identify:
- **Procurement Fraud**: Unusually high contract values or suspicious patterns
- **Process Irregularities**: Deviations from standard procurement procedures
- **Data Quality Issues**: Erroneous or inconsistent data entries
- **Policy Violations**: Contracts that don't comply with established guidelines
