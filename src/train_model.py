import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
import psycopg2
from sqlalchemy import create_engine
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Connect to the PostgreSQL database
connection = psycopg2.connect(
    dbname="lol_matchinfo",
    user="kai",
    password="kkaakkaa",
    host="localhost",
    port="5432"
)

engine = create_engine(
    'postgresql://kai:kkaakkaa@localhost:5432/lol_matchinfo'
)



# Data retrieval
data_query = 'SELECT * FROM fin;'
final_data = pd.read_sql_query(data_query, engine)

# Data preprocessing
final_data['participant_id'] = final_data['participant_id'].astype(str)
final_data['championName'] = final_data['championName'].astype(str)

# Feature Engineering
final_data['timestamp_mins'] = final_data['event_timestamp'] / 60000
final_data['KDA_Ratio'] = (final_data['kills'] + final_data['assists']) / (final_data['deaths'] + 1)
final_data['GPM'] = final_data['totalGold'] / (final_data['timestamp_mins'])

le_champion = LabelEncoder()
final_data['championName'] = le_champion.fit_transform(final_data['championName'])

le_position = LabelEncoder()
final_data['teamPosition'] = le_position.fit_transform(final_data['teamPosition'])


# Define features to scale
features_to_scale = ['damagePerMinute', 'totalDamageTaken', 'xp', 'kills', 'deaths', 'assists',
                      'event_timestamp', 'KDA_Ratio', 'GPM', 'turret_plates', 'inhibitor_kills', 'dragon_kills','killParticipation','teamDamagePercentage','teamGoldDifference']
print(final_data.shape)

# Extract features and target variable
X = final_data[features_to_scale]
y = final_data['win']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define feature weights
feature_weights = {
    'damagePerMinute': 1.4,
    'totalDamageTaken': 1.1,
    'xp': 1.2,
    'kills': 1.5,
    'deaths': 1.2,
    'assists': 1.1,
    'event_timestamp': 1.0,
    'KDA_Ratio': 1.2,
    'GPM': 1.3,
    'turret_plates': 1.3,
    'inhibitor_kills': 1.2,
    'dragon_kills': 1.1,
    'killParticipation':1.2
    ,'teamDamagePercentage':1.3,
    'teamGoldDifference':1.4
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

# Define custom scoring function for cross-validation
def custom_scoring_function(y_true, y_pred):
    print(f"Shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    return accuracy_score(y_true, y_pred)

custom_scorer = make_scorer(custom_scoring_function, greater_is_better=True)

# Models
models = [
    DecisionTreeClassifier(random_state=42),
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
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


def train_and_predict(model, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

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

    if "RandomForest" in model_name:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC: {roc_auc}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return accuracy, precision, recall, f1
# Individual models
results = Parallel(n_jobs=3)(
    delayed(train_and_predict)(model, X_train, y_train, X_test, y_test)
    for model in models
)


# Neural Network using TensorFlow/Keras
def create_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled_weighted.shape[1],)),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model_nn = KerasClassifier(model=create_model, epochs=15, batch_size=32, verbose=0)

model_nn.fit(X_train_scaled_weighted, y_train)

# Evaluate on the test set
y_pred_test = model_nn.predict(X_test_scaled_weighted)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print(f"Final Test Metrics:")
print(f"Accuracy: {accuracy_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1 Score: {f1_test}")

# AUC-ROC for Neural Network
y_pred_proba_nn = model_nn.predict_proba(X_test_scaled_weighted)[:, 1]
roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn)

# Plot ROC curve for Neural Network
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
plt.figure()
plt.plot(fpr_nn, tpr_nn, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_nn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Neural Network')
plt.legend(loc="lower right")
plt.show()


# Training Neural Network
print("Training Neural Network...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_nn.fit(X_train_scaled_weighted, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

keras_model = model_nn.model_
history = keras_model.history.history
print(history)


# Plotting training history
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


y_pred_test = model_nn.predict(X_test_scaled_weighted)
accuracy_nn = accuracy_score(y_test, y_pred_test)
print(f"Neural Network Accuracy: {accuracy_nn}")
engine.dispose()