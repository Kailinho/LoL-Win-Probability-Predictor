import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from joblib import Parallel, delayed
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
from joblib import dump

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DATABASE_URI = os.getenv("DATABASE_URI")

# Connect to the PostgreSQL database
connection = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

engine = create_engine(
    DATABASE_URI
)

# Data retrieval
data_query = 'SELECT * FROM finaldata;'
final_data = pd.read_sql_query(data_query, engine)

# Data preprocessing
final_data['participant_id'] = final_data['participant_id'].astype(str)
final_data['championName'] = final_data['championName'].astype(str)

le_champion = LabelEncoder()
final_data['championName'] = le_champion.fit_transform(final_data['championName'])

le_position = LabelEncoder()
final_data['teamPosition'] = le_position.fit_transform(final_data['teamPosition'])

le_team_id = LabelEncoder()
final_data['teamId'] = le_team_id.fit_transform(final_data['teamId'])


# Define features to scale
features_to_scale = ['damagePerMinute', 'totalDamageTaken', 'xp', 'kills', 'deaths', 'assists',
                      'KDA_Ratio', 'turret_plates', 'inhibitor_kills', 'dragon_kills','killParticipation','teamDamagePercentage','teamGoldDifference',
                      'laneGoldDifference','damagePerGold','laneExpDifference','visionScore']
print(final_data.shape)

# Extract features and target variable
X = final_data[features_to_scale]
y = final_data['win']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define feature weights
feature_weights = {
    'teamGoldDifference':1.4,
    'damagePerGold':1.4,
    'laneExpDifference':1.4,
    'teamDamagePercentage':1.3,
    'KDA_Ratio': 1.2,
    'inhibitor_kills': 1.35,
    'dragon_kills': 1.35,
    'xp': 1.3,
    'kills': 1.2,
    'laneGoldDifference':1.4,
    'deaths': 1.2,
    'turret_plates': 1.2,
    'damagePerMinute': 1.25,
    'visionScore': 1.1,
    'killParticipation':1.2,
    'totalDamageTaken': 1.1,
    'assists': 1.1
}

# Apply feature weights to training and test sets
X_train_weighted = X_train.copy()
X_test_weighted = X_test.copy()

for feature, weight in feature_weights.items():
    X_train_weighted[:, features_to_scale.index(feature)] *= weight
    X_test_weighted[:, features_to_scale.index(feature)] *= weight

# Scale the weighted features
scaler = StandardScaler()
X_train_scaled_weighted = scaler.fit_transform(X_train_weighted)
X_test_scaled_weighted = scaler.transform(X_test_weighted)

# Adjust the number of features after applying weights
num_original_features = len(features_to_scale)
num_weighted_features = len(feature_weights)
num_total_features = num_original_features + num_weighted_features

X_train_scaled_weighted = X_train_scaled_weighted[:, :num_total_features]
X_test_scaled_weighted = X_test_scaled_weighted[:, :num_total_features]


# Define custom scoring function for cross-validation
def custom_scoring_function(y_true, y_pred):
    print(f"Shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    return accuracy_score(y_true, y_pred)

custom_scorer = make_scorer(custom_scoring_function, greater_is_better=True)

# Models
models = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
]


# Ensemble model
ensemble_model = VotingClassifier(estimators=[(str(i), model) for i, model in enumerate(models)], voting='soft')

# Training the ensemble model
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Model Accuracy: {accuracy_ensemble}')

# Fit RandomForestClassifier separately
random_forest_model = models[1]
random_forest_model.fit(X_train, y_train)

# Function to train and predict for individual models
def train_and_predict(model, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    # For classifiers that support predict_proba, use the probabilities of the positive class (1)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    if y_pred_proba is not None:
        # Print additional metrics for probabilistic predictions
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC: {roc_auc}")

    return accuracy, precision, recall, f1, y_pred_proba

# Training and predicting for individual models in parallel
results = Parallel(n_jobs=3)(
    delayed(train_and_predict)(model, X_train, y_train, X_test, y_test)
    for model in models
)

# Save the best-performing models
dump(ensemble_model, 'best_ensemble_model.joblib')
dump(random_forest_model, 'best_random_forest_model.joblib')

# Closing the database connection
engine.dispose()