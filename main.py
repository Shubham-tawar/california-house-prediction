import os
import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt


MODEL_FILE = "model.pkl"
PIPELINE_FILE = 'pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    # For numerical columns
    num_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For categorical columns
    cat_pipline = Pipeline([ 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipline, num_attribs), 
        ('cat', cat_pipline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Lets train the model
    housing = pd.read_csv("/californiaHousing/housing.csv")

    # Create a stratified test set
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False) 
        housing = housing.loc[train_index].drop("income_cat", axis=1)  
    
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs) 
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained. Congrats!")
else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(r"C:\Users\Shubham Tawar\OneDrive\Desktop\coding\Python\californiaHousing\input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    # input_data['median_house_value'] = predictions --> this line is replacing the actual values with predicted values and saving a new csc file as  output later on
    ## now testing the model performance using root_mean_squared_error
    actual_house_vals= input_data['median_house_value']
    rmse = root_mean_squared_error(actual_house_vals, predictions)
    print(f"Root Mean Squared Error on the test set with Random Forest Regression is: {rmse}")    
    # now saving the predictions to a input csv file with a new column 'predicted_median_house_value'
    input_data.insert(len(input_data.columns), 'predicted_median_house_value', predictions)
    input_data.to_csv(r"C:\Users\Shubham Tawar\OneDrive\Desktop\coding\Python\californiaHousing\output.csv", index=False)
    
    #plot the predictions vs actual values
    

    plt.figure(figsize=(6,6))
    plt.scatter(actual_house_vals, predictions, alpha=0.5 )
    plt.plot([actual_house_vals.min(), actual_house_vals.max()],
            [actual_house_vals.min(), actual_house_vals.max()],
            'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Median House Value")
    plt.show()

    print("Inference is complete, results saved to output.csv Enjoy!")

    
