# Importing necessary libraries
import requests
import pandas as pd
import os
from sqlalchemy import create_engine
import time
import random
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

# List of PUUIDs for initial players
puuids =  [
  'olME8kPQW7QXINS1s21Pgu14wuQ5z7NI8NrPdtZ1HOaedPBehzRu8sxKzXwJI8zN_J1tc32FE4PlOQ',
  'BLjbInRTXR9j6qoB1FqWsr6Gmh-a5XcPIKR0_5ZeDUMbUZNXZIdvKJ4IVBbejBuJPKJyDhewn6NHMg',
  'VZJDfB6Ht-4PJvU1kuS5oytUDz-wYUv5eZba_Skj1owTP520sgbtB_wQj2pEq03EAtvHONJhJhtgHw',
  'KGczmH8uvHsO5KWrHpXo4RRc7mVeCD-3WmEAvuSXbl4Rngdw8Su71O--bTVjxe7VDBYThArQr8X23w',
  'GGqHV5vvm-hptVWuAy2Qp72AYRaEdfHn2dWc1XkgNA5M4VpGBsGirp4hj9fqn5RL8KqiSgX865k6wg',
  'OskKUKi6dAgulOucjX8V1o8Cwm_5g6ijdSeODUoZBUWOtCkjpzKKWN39Dcrjd8ZtTAM5nDmKVsgR2Q',
  'PoVfnBR3fMWydgoXHcj-Ew3OyoUrtpOockHHEC7_N6Fe4IdGz9KvJIZoGcWw_Od0zEZX46VDRdSJQw'
]



# Riot Games API base URL and API key
riot_api_base_url = 'https://americas.api.riotgames.com/'
api_key = os.getenv('RIOT_API_KEY')

# HTTP headers for API requests
headers = {'X-Riot-Token': api_key}

# Summoner name for API requests
summoner_name = 'Kailinho'

# Database connection details
database_uri = os.getenv('DATABASE_URI')
engine = create_engine(database_uri)

# DataFrames for storing player information, events, and total match statistics
participants_df = pd.DataFrame()
events_df = pd.DataFrame()
total_matchstats_df = pd.DataFrame()


# Function to retrieve matches for a given PUUID
def get_matches(puuid, requests_per_minute=50, cooldown_duration=30):
    """
    Retrieve and process matches for a given player (PUUID).
    
    :param puuid: Player's PUUID
    :param requests_per_minute: Number of API requests per minute (default: 50)
    :param cooldown_duration: Cooldown duration between requests in seconds (default: 30)
    """
    global participants_df, events_df, total_matchstats_df
    try:
        # Retrieve match IDs for the player
        match_list_url = f'{riot_api_base_url}lol/match/v5/matches/by-puuid/{puuid}/ids?start=1&count=50&api_key={api_key}'
        response = requests.get(match_list_url , headers=headers)
        match_id_list = response.json()

        print(f'Match IDs for PUUID {puuid}: {match_id_list}')

        # Iterate through match IDs and retrieve match info
        for match_id in match_id_list:
           get_match_info(match_id)
           time.sleep(60 / requests_per_minute)


    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')

    time.sleep(cooldown_duration)


# Function to retrieve additional players based on initial PUUIDs
def get_more_players(initial_puuids, num_additional_players=8):
    """
    Retrieve additional players based on the initial set of PUUIDs.
    
    :param initial_puuids: Initial set of PUUIDs
    :param num_additional_players: Number of additional players to retrieve (default: 8)
    :return: List of additional PUUIDs
    """
    additional_puuids = set()
    for puuid in initial_puuids:
        try:
            match_list_url = f'{riot_api_base_url}lol/match/v5/matches/by-puuid/{puuid}/ids?start=1&count=75&api_key={api_key}'
            response = requests.get(match_list_url, headers=headers)
            match_id_list = response.json()

            for match_id in random.sample(list(match_id_list), min(num_additional_players, len(match_id_list))):
                match_url = f'{riot_api_base_url}lol/match/v5/matches/{match_id}?api_key={api_key}'
                match_response = requests.get(match_url, headers=headers)
                match_data = match_response.json()

                participants = match_data.get('info', {}).get('participants', [])
                additional_puuids.update(participant['puuid'] for participant in participants)

        except requests.exceptions.RequestException as error:
            print(f'Error: {error}')

    return list(additional_puuids)

# Columns for participant data
participants_columns = ['match_id', 'participant_id', 'laneGoldDifference',
                        'totalDamageDone', 'totalDamageTaken', 'xp','damagePerGold',
                        'timeEnemySpentControlled', 'totalGold', 'timestamp', 'teamGoldDifference', 'teamPosition']


def collect_aggregated_data(frames, match_id,total_matchstats_df):
    """
    Collect and aggregate data from match frames.
    
    :param frames: List of match frames
    :param match_id: Match ID
    :param total_matchstats_df: DataFrame for total match statistics
    :return: Updated participants_df and events_df
    """    
    global participants_df, events_df

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
    append_to_database(participants_df, 'participants')
    append_to_database(events_df, 'events')
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
                     'totalUnitsHealed', 'turretKills', 'visionScore','teamId', 'win','championName','teamPosition']

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
        append_to_database(total_matchstats_df, 'total_matchstats')
        collect_aggregated_data(match_timeline,match_id,total_matchstats_df)

    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')


# Retrieve additional PUUIDs and combine with initial set
additional_puuids = get_more_players(puuids)
all_puuids = puuids + additional_puuids

# Iterate through all PUUIDs to retrieve matches
for puuid in all_puuids:
    get_matches(puuid)
