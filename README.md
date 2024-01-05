# League of Legends Win Probability Prediction

This project focuses on predicting the win probability of League of Legends (LoL) matches using machine learning models. The system comprises two main components: data collection (`match.py`) and model training (`train_model.py`).

# [Riot games does not provide real-time game data, so this application is no longer being worked on.]


## Data Collection (`match.py`)

The data collection module (`match.py`) interacts with the Riot Games API to gather information about LoL matches. It leverages the match ID obtained from a summoner's match history to fetch detailed match data and timeline information. The collected data includes participant statistics, events, and aggregated match statistics. The information is then stored in a PostgreSQL database.

### Setup

1. Replace `api_key` with your Riot Games API key.
2. Set `puuids` based on the puuids gathered (`api.js`) for interested users.
3. Ensure PostgreSQL is installed and running locally.
4. Update `database_uri` with your PostgreSQL database connection URI.

### Usage

Run `python match.py` to start collecting match data.

## Model Training (`train_model.py`)

The model training module (`train_model.py`) utilizes machine learning models to predict LoL match outcomes. It connects to the PostgreSQL database, retrieves preprocessed data, and trains an ensemble model comprising Decision Trees, Neural Networks, and Random Forests. The models are evaluated using a custom scoring function that emphasizes specific features.

## Results

Ensemble Model Accuracy: **0.8995223684690269**

DecisionTreeClassifier Metrics:
Accuracy: **0.8994713238779305**

Precision: **0.9028663526600252**

Recall: **0.9053317915209608**

F1 Score: **0.9040973913043479**

AUC-ROC: **0.8991835370591736**



RandomForestClassifier Metrics:
Accuracy: **0.945046851642542**

Precision: **0.9447282796815507**

Recall: **0.9506248519720802**

F1 Score: **0.9476673935084236**

AUC-ROC: **0.9896320727295794**


### Setup

1. Update the PostgreSQL connection details in the `train_model.py` script.
2. Ensure required Python libraries are installed .

### Usage

Run `python train_model.py` to train and evaluate the machine learning models.

## Requirements

- Python 3.x
- PostgreSQL
- Riot Games API key

## Acknowledgments

- This project utilizes data from the [Riot Games API](https://developer.riotgames.com/).
- The machine learning models are implemented using [scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/), and [Keras](https://keras.io/).


Feel free to contribute, raise issues, or suggest improvements!
