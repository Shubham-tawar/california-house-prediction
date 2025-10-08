# 🏡 California Housing Price Prediction

This project uses **Machine Learning** to predict median house values in California districts based on the famous **California Housing dataset**.  
It demonstrates the full end-to-end ML workflow — from data preprocessing and feature engineering to model training, evaluation, and deployment.

---

## 📂 Project Structure

californiaHousing/
│
├── housing.csv # Original dataset
├── input.csv # Input data for prediction (test split)
├── output.csv # Model predictions
├── model.pkl # Trained Random Forest model (saved via joblib) ->not included in the repo, your .pkl file will be created once you run the code
├── pipeline.pkl # Preprocessing pipeline (saved via joblib) ->not included in the repo, your .pkl file will be created once you run the code
├── main.py # Main training + inference script
|-- python.py # initial model testing and selection code for different regression algos, you can add your models to test the best accuracy
└── README.md # Project documentation



---

## 🚀 Features

- **Data Preprocessing** using `ColumnTransformer` and `Pipeline`
- **Feature Engineering**:  
  Adds powerful derived attributes to improve model accuracy:
  
- **Model Training** using `RandomForestRegressor`
- **Cross-validation** for robust evaluation
- **Joblib-based Model Saving/Loading**
- **Automatic Inference** pipeline to predict new data from `input.csv`
- **RMSE Evaluation** to measure accuracy

---
## How It Works
### Training Mode

If the model doesn’t exist yet (model.pkl not found), the script:

Loads housing.csv

Performs stratified train-test split

Preprocesses the data using pipelines

Engineers additional meaningful features

Trains a Random Forest Regressor

Saves the model and preprocessing pipeline

### Inference Mode

If model.pkl already exists:

Loads input.csv

Transforms the data using the saved pipeline

Generates predictions using the trained model

Saves results to output.csv

---

##RMSE (Root Mean Squared Error) 
measures how much predicted prices deviate from actual ones.
--**Typical results:**
--**Before feature engineering**: ~47,000
--After feature engineering: ~35,000 – 40,000
--Lower RMSE = Better predictions ✅

---

##Future Improvements
--🔧 Hyperparameter tuning with GridSearchCV

--🧠 Try advanced models like XGBoost or LightGBM

--🌐 Deploy as an API or Streamlit web app

--📦 Containerize with Docker for reproducibility
-- Can use Cross_val_score to train the model for better accuracy 

---
## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/california-housing-ml.git
cd california-housing-ml
---


