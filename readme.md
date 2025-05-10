for some reason the virtual environment is ignored by git, therefore you will need to pip install the libraries yourself

files and their responsability

data-preview-ted -> csv with a subset from ted's data.europa.eu that will be used to train the model
isolation_forest.py -> reads and transforms the data, trains the model on the data, saves the model to isolation_forest_model.pkl
isolation_forest_model.pkl -> saved model after training
isolation_forest_predict.py -> loads the trained model and can predict either the fed data is an outlier or not based on the loaded model
results.csv -> results from the fed data in bulk
outlier_distributions.png -> showcases the distributions of all the outliers within the dataset
outlier_scatter_mix.png -> showcases something

usage:

1.Train the model with a dataset using isolation_forest.py
2.Find the outliers within a csv, or hardwired into the script using isolation_forest_predict.py
3.Visualize the outcome

Usage of isolation_forest_predict.py
    Process a CSV file containing new records:
    ```
    python procurement_outlier_detection_usage.py --csv new_procurements.csv results.csv
    ```
    Process a single record directly from command line:
    ```
    python procurement_outlier_detection_usage.py --record VALUE_EURO=5000000 CPV=45000000 CRIT_PRICE_WEIGHT=60
    ```
    Outcome is available within the console and results.csv











# Process a CSV file (basic usage)
python improved_isolation_forest_predict.py --csv your_input.csv your_output.csv

# Process a CSV file with custom model file
python improved_isolation_forest_predict.py --csv your_input.csv your_output.csv your_model.pkl

# Process a single record
python improved_isolation_forest_predict.py --record feature1=value1 feature2=value2