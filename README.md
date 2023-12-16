# League of Legends Win Probability Prediction

This project focuses on predicting the win probability of League of Legends (LoL) matches using machine learning models. The system comprises two main components: data collection (`match.py`) and model training (`train_model.py`).

## Data Collection (`match.py`)

The data collection module (`match.py`) interacts with the Riot Games API to gather information about LoL matches. It leverages the match ID obtained from a summoner's match history to fetch detailed match data and timeline information. The collected data includes participant statistics, events, and aggregated match statistics. The information is then stored in a PostgreSQL database.

### Setup

1. Replace `api_key` with your Riot Games API key.
2. Set `summoner_name` to the summoner name of interest.
3. Ensure PostgreSQL is installed and running locally.
4. Update `database_uri` with your PostgreSQL database connection URI.

### Usage

Run `python match.py` to start collecting match data.

## Model Training (`train_model.py`)

The model training module (`train_model.py`) utilizes machine learning models to predict LoL match outcomes. It connects to the PostgreSQL database, retrieves preprocessed data, and trains an ensemble model comprising Decision Trees, Neural Networks, and Random Forests. The models are evaluated using a custom scoring function that emphasizes specific features.

### Setup

1. Update the PostgreSQL connection details in the `train_model.py` script.
2. Ensure required Python libraries are installed (`pandas`, `scikit-learn`, `tensorflow`).

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
