import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # to combine numerical and categorical pipelines
from sklearn.impute import SimpleImputer # to handle missing values
# For scaling and encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# To train the data on various models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error # to evaluate the model error
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal
from sklearn.model_selection import cross_val_score # for cross-validation to avoid overfitting of data


# 1. Load the data
housing = pd.read_csv(r"C:\Users\Shubham Tawar\OneDrive\Desktop\coding\Python\californiaHousing\housing.csv")

# 2. Create a stratified test set based on income category attribute
# stratification is done to ensure that the test set is representative of the overall distribution of the data, i.e the income categories in the test set should reflect those in the entire dataset.
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)

#7. Train and evaluate models Decision Tree
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Evaluate each model using RMSE
for model_name, model in models.items():
    model.fit(housing_prepared, housing_labels) # Train the model
    housing_predictions = model.predict(housing_prepared) # Make predictions
    rmse = root_mean_squared_error(housing_labels, housing_predictions) # Calculate RMSE     
    print(f"{model_name} RMSE: {rmse}")
    # got overfitting of Decision Tree, now we will do cross-validation to get better estimate of the model performance
    cross_rmses = -cross_val_score(
        model,
        housing_prepared,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10)
    print(pd.Series(cross_rmses).describe())


# Random Forest generally performs the best among these models.
# 8. Final evaluation on the test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_model = models["Random Forest"]
final_predictions = final_model.predict(X_test_prepared)
final_rmse = root_mean_squared_error(y_test, final_predictions)

print(f"Final Random Forest RMSE on Test Set: {final_rmse}")
# The final RMSE gives an estimate of how well the model is expected to perform on unseen data.
# The income category attribute is created by dividing the median_income attribute into 5 categories using pd.cut(). This helps in stratified sampling to ensure that each income category is proportionally represented in both the training and test sets.
# StratifiedShuffleSplit is used to create a train-test split that maintains the distribution of the income categories in both sets.
# The full_pipeline combines both numerical and categorical pipelines using ColumnTransformer, allowing for simultaneous preprocessing of different types of data.
# The models dictionary allows for easy experimentation with different regression algorithms. Each model is trained and evaluated
# using RMSE to compare their performance.
# The final evaluation on the test set provides an unbiased estimate of the model's performance on new, unseen data.
# This comprehensive approach ensures that the model is robust and generalizes well to real-world scenarios.
# The income category attribute is created by dividing the median_income attribute into 5 categories using pd.cut(). This helps in stratified sampling to ensure that each income category is proportionally represented in both the training and test sets.
# StratifiedShuffleSplit is used to create a train-test split that maintains the distribution of the
# income categories in both sets.
# The full_pipeline combines both numerical and categorical pipelines using ColumnTransformer, allowing for simultaneous preprocessing of
# different types of data.
# The models dictionary allows for easy experimentation with different regression algorithms. Each model is trained and evaluated
# using RMSE to compare their performance.
# The final evaluation on the test set provides an unbiased estimate of the model's performance on new
# unseen data.
# This comprehensive approach ensures that the model is robust and generalizes well to real-world scenarios.
