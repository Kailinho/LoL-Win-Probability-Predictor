# Importing necessary libraries
import requests
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from joblib import load
import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Loading environment variables
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

# Riot Games API base URL and API key
riot_api_base_url = 'https://americas.api.riotgames.com/'
api_key = os.getenv('RIOT_API_KEY')

# HTTP headers for API requests
headers = {'X-Riot-Token': api_key}

# Database connection details
database_uri = os.getenv('DATABASE_URI')
engine = create_engine(database_uri)

# DataFrames for storing player information, events, and total match statistics
participants_df = pd.DataFrame()
events_df = pd.DataFrame()
total_matchstats_df = pd.DataFrame()

def collect_aggregated_data(frames, match_id,total_matchstats_df):
    """
    Collect and aggregate data from match frames.
    
    :param frames: List of match frames
    :param match_id: Match ID
    :param total_matchstats_df: DataFrame for total match statistics
    :return: Updated participants_df and events_df
    """    
    global participants_df, events_df

    participants_columns = ['match_id', 'participant_id', 'laneGoldDifference',
                            'totalDamageDone', 'totalDamageTaken', 'xp','damagePerGold',
                            'timeEnemySpentControlled', 'totalGold', 'timestamp', 'teamGoldDifference', 'teamPosition']
    events_columns = ['match_id', 'participant_id', 'kills', 'deaths', 'assists', 'dragon_kills', 'turret_plates',
                      'inhibitor_kills']
    events_df = pd.DataFrame(columns=events_columns)
    events_data = {'match_id': {}, 'participant_id': {}, 'kills': {}, 'deaths': {}, 'assists': {}, 'timestamp': {},
                   'dragon_kills': {}, 'turret_plates': {}, 'inhibitor_kills': {}, }
    
    # Iterate through match frames, each frame is per minute 
    for frame in frames:
        #Retreive participant and event frames to grab different data later
        participant_frames = frame['participantFrames']
        events = frame['events']

        participant_data = {col: {} for col in participants_columns}
        timestamp = frame['timestamp']
        team_gold = {'team1': 0, 'team2': 0}

        # Process events in the frame to grab features - kills, deaths, assists, dragons, turret plates, inhibitors
        for event in events:
            if event['type'] == 'CHAMPION_KILL':
                # Increment kill count for killer
                if 'killerId' in event:
                    killer_id = event['killerId']
                    events_data['kills'][killer_id] = events_data['kills'].get(killer_id, 0) + 1
                    events_data['participant_id'][killer_id] = killer_id
                    events_data['match_id'][killer_id] = match_id
                    events_data['timestamp'][killer_id] = event['timestamp']
                # Increment death count for victim
                if 'victimId' in event:
                    victim_id = event['victimId']
                    events_data['deaths'][victim_id] = events_data['deaths'].get(victim_id, 0) + 1
                    events_data['participant_id'][victim_id] = victim_id
                    events_data['match_id'][victim_id] = match_id
                    events_data['timestamp'][victim_id] = event['timestamp']
                # Increment assist count for assisting participants
                if 'assistingParticipantIds' in event:
                    for assist_id in event['assistingParticipantIds']:
                        events_data['assists'][assist_id] = events_data['assists'].get(assist_id, 0) + 1
                        events_data['participant_id'][assist_id] = assist_id
                        events_data['match_id'][assist_id] = match_id
                        events_data['timestamp'][assist_id] = event['timestamp']


            #Increment dragon kill count for each member of the team of the dragon's killer
            elif event['type'] == 'ELITE_MONSTER_KILL' and event['monsterType'] == 'DRAGON':
                team_id = event['killerTeamId']
                if team_id == 200 and 6 <= participant_id <= 10:
                    for teammate_id in range(6, 11):
                        events_data['dragon_kills'][teammate_id] = events_data['dragon_kills'].get(teammate_id, 0) + 1
                elif team_id == 100 and 1 <= participant_id <= 5:
                    for teammate_id in range(1, 6):
                        events_data['dragon_kills'][teammate_id] = events_data['dragon_kills'].get(teammate_id, 0) + 1
            #Increment turret plate count for participants that get a turret plate
            elif event['type'] == 'TURRET_PLATE_DESTROYED':
                killer_id = event['killerId']
                turret_plates_destroyed = events_data['turret_plates'].get(killer_id, 0) + 1
                events_data['turret_plates'][killer_id] = turret_plates_destroyed

            #Increment inhibitor count for each member of the team of the dragon's killer
            elif event['type'] == 'BUILDING_KILL' and event['buildingType'] == 'INHIBITOR_BUILDING':
                participant_id = event['killerId']
                if 6 <= participant_id <= 10:
                    for teammate_id in range(6, 11):
                        inhibitor_kills = events_data['inhibitor_kills'].get(teammate_id, 0) + 1
                        events_data['inhibitor_kills'][teammate_id] = inhibitor_kills
                elif 1 <= participant_id <= 5:
                    for teammate_id in range(1, 6):
                        inhibitor_kills = events_data['inhibitor_kills'].get(teammate_id, 0) + 1
                        events_data['inhibitor_kills'][teammate_id] = inhibitor_kills


        # Extract participant information
        for participant_id, participant_frame in participant_frames.items():
            try:
                participant_id = int(participant_id)
                total_damage_done = participant_frame['damageStats']['totalDamageDoneToChampions']
                total_damage_taken = participant_frame['damageStats']['totalDamageTaken']
                xp = participant_frame['xp']
                time_enemy_spent_controlled = participant_frame['timeEnemySpentControlled']
                total_gold = participant_frame['totalGold']
                if 1 <= participant_id <= 5:
                    team_gold['team1'] += total_gold
                elif 6 <= participant_id <= 10:
                    team_gold['team2'] += total_gold

                participant_id = int(participant_id)
                # Update data dictionary
                participant_data['match_id'][participant_id] = match_id
                participant_data['participant_id'][participant_id] = participant_id
                participant_data['totalDamageDone'][participant_id] = total_damage_done
                participant_data['totalDamageTaken'][participant_id] = total_damage_taken
                participant_data['xp'][participant_id] = xp
                participant_data['timeEnemySpentControlled'][participant_id] = time_enemy_spent_controlled
                participant_data['totalGold'][participant_id] = total_gold
                participant_data['timestamp'][participant_id] = timestamp           
                participant_data['teamPosition'][participant_id] = total_matchstats_df['teamPosition'][participant_id-1]
                participant_data['damagePerGold'][participant_id] = total_damage_done/total_gold
            except Exception as e:
                print(f'Error: {e}. Check the structure of participant_df and participant_frame.')
        
        for participant_id, _ in participant_frames.items():
            try:
                participant_id = int(participant_id)
                participant_data['teamGoldDifference'][participant_id] = (
                    team_gold['team1'] - team_gold['team2']
                    if 1 <= participant_id <= 5
                    else team_gold['team2'] - team_gold['team1']
                )

            except KeyError as e:
                print(f'Error: {e}. Check the structure of participant_frame.')

        #Update dataframes for participants and events with data from current frame
        participants_df = pd.concat([participants_df, pd.DataFrame(participant_data)], ignore_index=True).fillna(0)
        events_df = pd.concat([events_df, pd.DataFrame(events_data)],ignore_index=True).fillna(0)
    #Append final dataframe to SQL database
    # append_to_database(participants_df, 'live_participants')
    # append_to_database(events_df, 'live_events')
    return participants_df, events_df

def append_to_database(dataframe, table_name):
    """
    Append a DataFrame to the specified database table.
    
    :param dataframe: DataFrame to append
    :param table_name: Name of the database table
    """    
    try:
        dataframe.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Data successfully appended to {table_name} in the database.")
    except Exception as e:
        print(f"Error: {str(e)}")

def get_match_info(match_id):
    """
    Retrieve and process match information for a given match ID.
    
    :param match_id: Match ID
    """    
    global total_matchstats_df
    try:
        # Retrieve match timeline data
        match_timeline_url = f'{riot_api_base_url}lol/match/v5/matches/{match_id}/timeline?api_key={api_key}'
        match_timeline_response = requests.get(match_timeline_url, headers=headers)
        match_timeline_data = match_timeline_response.json()

        if 'info' not in match_timeline_data:
            print('Error: No "info" in match timeline data')
            return

        match_url = f'{riot_api_base_url}lol/match/v5/matches/{match_id}?api_key={api_key}'
        match_response = requests.get(match_url, headers=headers)
        
        # Check if match data is present
        match_data = match_response.json()
        if match_data is None:
            print('Error: Failed to fetch match data')
            return

        # Check if 'metadata' key is present in match_data
        if 'metadata' not in match_data:
            print('Error: No "metadata" in match data')
            return

        match_metadata = match_data['metadata']
        match_info = match_data['info']

        # Check if the mapId is for Summoner's Rift (mapId 11)
        if match_info.get('mapId', 0) != 11:
            return

        match_timeline = match_timeline_data['info'].get('frames', [])
    
        # Additional information for each participant from key_list
        key_list = ['pentaKills', 'timeCCingOthers', 'totalMinionsKilled',
                     'totalUnitsHealed', 'turretKills', 'visionScore','teamId', 'championName','teamPosition']

        participants_data = []

        for participant_id in range(1, 11):
            participant_data = {'participant_id': participant_id}
            for key in key_list:
                # Extract additional participant information from key_list
                if key in match_info['participants'][participant_id - 1]:
                    participant_data[key] = match_info['participants'][participant_id - 1][key]

            # Add additional participant information to participants_data list
            participants_data.append(participant_data)

        key_list_with_challenge_stats = key_list + ['killParticipation', 'teamDamagePercentage']
        total_matchstats_df = pd.DataFrame(columns=key_list_with_challenge_stats)
    
        # Concatenate participants_data into total_matchstats_df
        total_matchstats_df = pd.concat([total_matchstats_df, pd.DataFrame(participants_data)], ignore_index=True).fillna(0)

        # Process additional challenge statistics
        for participant_id in range(1, 11):
            challenges = match_info['participants'][participant_id - 1].get('challenges', {})  # Get the challenges dictionary or an empty dictionary if it doesn't exist
            kill_participation = challenges.get('killParticipation', 0)  # Get the value for 'killParticipation' or default to 0 if it doesn't exist
            team_damage_percentage = challenges.get('teamDamagePercentage', 0)  # Get the value for 'teamDamagePercentage' or default to 0 if it doesn't exist

            # Update the DataFrame with the values
            total_matchstats_df.loc[total_matchstats_df['participant_id'] == participant_id, 'killParticipation'] = kill_participation
            total_matchstats_df.loc[total_matchstats_df['participant_id'] == participant_id, 'teamDamagePercentage'] = team_damage_percentage

        # Type conversion for necessary columns       
        total_matchstats_df[[ 'pentaKills', 
                 'timeCCingOthers', 'totalMinionsKilled',
                  'totalUnitsHealed', 'turretKills', 'visionScore' ]] = total_matchstats_df[['pentaKills', 
                 'timeCCingOthers', 'totalMinionsKilled',
                  'totalUnitsHealed', 'turretKills', 'visionScore']].astype(int)
        total_matchstats_df['match_id'] = match_metadata.get('matchId', 'N/A')
        total_matchstats_df['gameDuration'] = match_info['gameDuration']


        # Append total_matchstats_df to the database and collect aggregated data
        # append_to_database(total_matchstats_df, 'live_total_matchstats')
        collect_aggregated_data(match_timeline,match_id,total_matchstats_df)

    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')

get_match_info("NA1_4862171636")



# Data retrieval for training data
training_data_query = 'SELECT * FROM finaldata;'
training_data = pd.read_sql_query(training_data_query, engine)

# Data preprocessing for training data
training_data['participant_id'] = training_data['participant_id'].astype(str)
training_data['championName'] = training_data['championName'].astype(str)

le_champion = LabelEncoder()
training_data['championName'] = le_champion.fit_transform(training_data['championName'])

le_position = LabelEncoder()
training_data['teamPosition'] = le_position.fit_transform(training_data['teamPosition'])

le_team_id = LabelEncoder()
training_data['teamId'] = le_team_id.fit_transform(training_data['teamId'])




ensemble_model = load('best_ensemble_model.joblib')
random_forest_model = load('best_random_forest_model.joblib')

# Data retrieval for new data
data_query = '''
    WITH RankedData AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY participant_id ORDER BY timestamp DESC) AS row_num
        FROM
            matchdata
    )
    SELECT
        *
    FROM
        RankedData
    WHERE
        row_num = 1;
'''

new_data = pd.read_sql_query(data_query, engine)

# Data preprocessing for new data
new_data['participant_id'] = new_data['participant_id'].astype(str)
new_data['championName'] = new_data['championName'].astype(str)

le_champion = LabelEncoder()
new_data['championName'] = le_champion.fit_transform(new_data['championName'])

le_position = LabelEncoder()
new_data['teamPosition'] = le_position.fit_transform(new_data['teamPosition'])

le_team_id = LabelEncoder()
new_data['teamId'] = le_team_id.fit_transform(new_data['teamId'])

feature_weights = {
    'teamGoldDifference':1.4,
    'damagePerGold':1.4,
    'laneExpDifference':1.4,
    'teamDamagePercentage':1.3,
    'KDA_Ratio': 1.2,
    'inhibitor_kills': 1.3,
    'dragon_kills': 1.3,
    'xp': 1.3,
    'kills': 1.2,
    'laneGoldDifference':1.4,
    'deaths': 1.2,
    'turret_plates': 1.2,
    'damagePerMinute': 1.2,
    'visionScore': 1.2,
    'killParticipation':1.2,
    'totalDamageTaken': 1.1,
    'assists': 1.1
}
numerical_features = ['damagePerMinute', 'totalDamageTaken', 'xp', 'kills', 'deaths', 'assists',
                      'KDA_Ratio', 'turret_plates', 'inhibitor_kills', 'dragon_kills','killParticipation','teamDamagePercentage','teamGoldDifference',
                      'laneGoldDifference','damagePerGold','laneExpDifference','visionScore']
# Apply feature weights to the new data
new_data_weighted = new_data.copy()

for feature, weight in feature_weights.items():
    if feature in numerical_features:
        new_data_weighted[feature] *= weight

# Scale the weighted features
scaler = StandardScaler()
new_data_scaled_weighted = scaler.fit_transform(new_data_weighted[numerical_features])

# Make predictions using the loaded models
# For ensemble model
y_prob_ensemble = ensemble_model.predict_proba(new_data_scaled_weighted)[:, 1]

# For Random Forest model
y_prob_random_forest = random_forest_model.predict_proba(new_data_scaled_weighted)[:, 1]

# Print or use the probabilities as needed
new_data['ensemble_Probability'] = y_prob_ensemble
new_data['randomForest_Probability'] = y_prob_random_forest

# Print or use the probabilities as needed
print(new_data[['participant_id', 'ensemble_Probability', 'randomForest_Probability']])

new_data['team_color'] = new_data['teamId'].map({0: 'Blue', 1: 'Red'})
team_probabilities = new_data.groupby('team_color')[['ensemble_Probability', 'randomForest_Probability']].mean().reset_index()
print(team_probabilities)
# Close the database connection
engine.dispose()
