import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import psycopg2
from sqlalchemy import create_engine
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


# Connect to the PostgreSQL database
connection = psycopg2.connect(
    dbname="lol_matchinfo",
    user="kai",
    password="kkaakkaa",
    host="localhost",
    port="5432"
)

# Create a database engine for SQLalchemy
engine = create_engine(
    'postgresql://kai:kkaakkaa@localhost:5432/lol_matchinfo'
)

# Replace 'your_table_name' with your actual table names
participants_query = 'SELECT * FROM participants'
events_query = 'SELECT * FROM events'
complete_match_query = 'SELECT * FROM total_matchstats'

# Load data into DataFrames
participants_data = pd.read_sql_query(participants_query, connection)
events_data = pd.read_sql_query(events_query, connection)
complete_match_data = pd.read_sql_query(complete_match_query, connection)

participants_data = participants_data.drop_duplicates()
events_data = events_data.drop_duplicates()
complete_match_data = complete_match_data.drop_duplicates()


participants_data['participant_id'] = participants_data['participant_id'].astype(str)
participants_data['match_id'] = participants_data['match_id'].astype(str)

events_data['participant_id'] = events_data['participant_id'].astype(str)
events_data['match_id'] = events_data['match_id'].astype(str)
merged_data = pd.merge(participants_data, events_data, on=['participant_id','match_id'])

final_data = pd.merge(merged_data, complete_match_data[['match_id', 'win']], left_on='match_id', right_on='match_id')



# Assuming you have the features (X) and target variable (y) columns selected
# Replace 'feature1', 'feature2', ..., 'featureN' with your actual feature columns
X = final_data[['currentGold', 'magicDamageDone', 'physicalDamageDone', 'trueDamageDone', 'magicDamageTaken', 'physicalDamageTaken', 'trueDamageTaken', 'xp', 'timeEnemySpentControlled', 'totalGold']]
y = final_data['win']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Custom scoring function giving more weight to specific features
def custom_scoring_function(y_true, y_pred):
    print(f"Shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    weights = np.array([1.0, 1.0, 1.0, 1.02, 1.02, 1.0, 1.0, 1.0, 1.02])  # Adjust weights as needed
    weighted_predictions = y_pred * weights
    return accuracy_score(y_true, y_pred)

custom_scorer = make_scorer(custom_scoring_function, greater_is_better=True)


# Models
models = [
    DecisionTreeClassifier(random_state=42),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
]


# Ensemble model
# Ensemble model
ensemble_model = VotingClassifier(estimators=[(str(i), model) for i, model in enumerate(models)], voting='hard')

# Cross-validation
# cv_results = cross_val_score(ensemble_model, X, y, cv=5, scoring=custom_scorer)
# print(f'Ensemble Model Cross-Validation Results: {cv_results}')
# print(f'Mean Accuracy: {np.mean(cv_results)}')

# Training the ensemble model
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Model Accuracy: {accuracy_ensemble}')

batch_size = 1024
# Training the model without parallelism
# Training the models
for model in models:
    model_name = model.__class__.__name__
    print(f"Training {model_name}...")

    if "NeuralNetwork" in model_name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif "RandomForest" in model_name:
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            model.fit(X_train[i:end], y_train[i:end])
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")

# For Neural Network using TensorFlow/Keras
model_nn = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(1, activation='sigmoid')
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training Neural Network...")
model_nn.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the Neural Network
accuracy_nn = model_nn.evaluate(X_test_scaled, y_test)[1]
print(f"Neural Network Accuracy: {accuracy_nn}")

# Close the database connection
connection.close()
