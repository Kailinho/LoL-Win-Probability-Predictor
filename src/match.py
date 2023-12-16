import requests
import pandas as pd
import os
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine

riot_api_base_url = 'https://americas.api.riotgames.com/'
api_key = 'RGAPI-58285aa9-5170-4565-aab3-2d18a5968b3d'  # Replace with your actual API key
headers = {'X-Riot-Token': api_key}
summoner_name = 'Kailinho'
puuid = '-23srJpsUju9lhDerocvX4vyCUBo1hy_7vmUK-4Lrd8SWwvQhKNXXrN59DzJCfe7fDuwwJ96HoPx-A'
match_list_url = f'{riot_api_base_url}lol/match/v5/matches/by-puuid/{puuid}/ids?start=1&count=50&api_key={api_key}'
database_uri = 'postgresql://kai:kkaakkaa@localhost:5432/lol_matchinfo'
engine = create_engine(database_uri)
participants_df = pd.DataFrame()
events_df = pd.DataFrame()
total_matchstats_df = pd.DataFrame()

def get_matches():
    global participants_df, events_df, total_matchstats_df
    try:
        response = requests.get(match_list_url, headers=headers)
        match_id_list = response.json()

        print('Match IDs:', match_id_list)

        for match_id in match_id_list:
            get_match_info(match_id)

    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')


def collect_aggregated_data(frames,match_id):
    global participants_df, events_df 
    dataframes_to_save = {
        'Participants Data': participants_df,
        'Events Data': events_df
    }
    participants_columns = ['match_id','participant_id','currentGold', 'magicDamageDone', 'physicalDamageDone', 'trueDamageDone',
                             'magicDamageTaken', 'physicalDamageTaken', 'trueDamageTaken', 'xp',
                             'timeEnemySpentControlled', 'totalGold','timestamp']

    events_columns = ['match_id','participant_id', 'kills', 'deaths', 'assists']

    participants_df = pd.DataFrame(columns=participants_columns)
    events_df = pd.DataFrame(columns=events_columns)
    events_data = {'match_id': {}, 'participant_id': {}, 'kills': {}, 'deaths': {}, 'assists': {}, 'timestamp':{}}

    for frame in frames:
        participant_frames = frame['participantFrames']
        events = frame['events']
        participant_data = {col: {} for col in participants_columns}
        timestamp = frame['timestamp']
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
        
        
        # Extract participant information
        for participant_id, participant_frame in participant_frames.items():
            try:
                participant_id = int(participant_id)
                current_gold = participant_frame['currentGold']
                magic_damage_done = participant_frame['damageStats']['magicDamageDone']
                physical_damage_done = participant_frame['damageStats']['physicalDamageDone']
                true_damage_done = participant_frame['damageStats']['trueDamageDone']
                magic_damage_taken = participant_frame['damageStats']['magicDamageTaken']
                physical_damage_taken = participant_frame['damageStats']['physicalDamageTaken']
                true_damage_taken = participant_frame['damageStats']['trueDamageTaken']
                xp = participant_frame['xp']
                time_enemy_spent_controlled = participant_frame['timeEnemySpentControlled']
                total_gold = participant_frame['totalGold']

                # Update data dictionary
                participant_data['match_id'][participant_id] = match_id
                participant_data['participant_id'][participant_id] = participant_id
                participant_data['currentGold'][participant_id] = current_gold
                participant_data['magicDamageDone'][participant_id] = magic_damage_done
                participant_data['physicalDamageDone'][participant_id] = physical_damage_done
                participant_data['trueDamageDone'][participant_id] = true_damage_done
                participant_data['magicDamageTaken'][participant_id] = magic_damage_taken
                participant_data['physicalDamageTaken'][participant_id] = physical_damage_taken
                participant_data['trueDamageTaken'][participant_id] = true_damage_taken
                participant_data['xp'][participant_id] = xp
                participant_data['timeEnemySpentControlled'][participant_id] = time_enemy_spent_controlled
                participant_data['totalGold'][participant_id] = total_gold
                participant_data['timestamp'][participant_id] = timestamp

            except KeyError as e:
                print(f'Error: {e}. Check the structure of participant_frame and championStats.')

        participants_df = pd.concat([participants_df, pd.DataFrame(participant_data)], ignore_index=True).fillna(0)
        events_df = pd.concat([events_df, pd.DataFrame(events_data)],ignore_index=True).fillna(0)
    append_to_database(participants_df, 'participants')
    append_to_database(events_df, 'events')

    print('Participants Data:')
    print(participants_df)

    print('Events Data:')
    print(events_df)

    return participants_df, events_df
def append_to_database(dataframe, table_name):
    try:
        # Append DataFrame to the specified table in the database
        dataframe.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Data successfully appended to {table_name} in the database.")
    except Exception as e:
        print(f"Error: {str(e)}")


def get_match_info(match_id):
    global total_matchstats_df

    print(f'Fetching match details for match {match_id}...')
    try:
        match_timeline_url = f'{riot_api_base_url}lol/match/v5/matches/{match_id}/timeline?api_key={api_key}'
        match_timeline_response = requests.get(match_timeline_url, headers=headers)
        match_timeline_data = match_timeline_response.json()

        if 'info' not in match_timeline_data:
            print('Error: No "info" in match timeline data')
            return

        match_url = f'{riot_api_base_url}lol/match/v5/matches/{match_id}?api_key={api_key}'
        match_response = requests.get(match_url, headers=headers)
        match_data = match_response.json()

        if match_data is None:
            print('Error: Failed to fetch match data')
            return

        match_metadata = match_data['metadata']
        match_info = match_data['info']

        if match_info.get('mapId', 0) != 11:
            print('Error: This match is not on Summoner\'s Rift')
            return

        match_timeline = match_timeline_data['info'].get('frames', [])
        
        # Initialize match_stats_data as an empty dictionary
        match_stats_data = {}

        # Collect aggregated data
        collect_aggregated_data(match_timeline,match_id)

        # Additional information for each participant from key_list
        key_list = ['largestMultiKill', 'longestTimeSpentLiving', 'pentaKills', 'quadraKills',
                    'sightWardsBoughtInGame', 'timeCCingOthers', 'totalMinionsKilled',
                    'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'tripleKills', 'turretKills', 'visionScore', 'win']

        participants_data = []

        for participant_id in range(1, 11):  # Assuming there are 10 participants in a match
            participant_data = {'participantId': participant_id}
            for key in key_list:
                # Extract additional participant information from key_list
                if key in match_info['participants'][participant_id - 1]:
                    participant_data[key] = match_info['participants'][participant_id - 1][key]

            # Add additional participant information to participants_data list
            participants_data.append(participant_data)

        # match_stats_data['participantsData'] = participants_data
        total_matchstats_df = pd.DataFrame(columns=key_list)
        total_matchstats_df = pd.concat([total_matchstats_df, pd.DataFrame(participants_data)], ignore_index=True).fillna(0)
        total_matchstats_df[['largestMultiKill', 'longestTimeSpentLiving', 'pentaKills', 'quadraKills', 'sightWardsBoughtInGame', 'timeCCingOthers', 'totalMinionsKilled', 'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'tripleKills', 'turretKills', 'visionScore']] = total_matchstats_df[['largestMultiKill', 'longestTimeSpentLiving', 'pentaKills', 'quadraKills', 'sightWardsBoughtInGame', 'timeCCingOthers', 'totalMinionsKilled', 'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'tripleKills', 'turretKills', 'visionScore']].astype(int)
        total_matchstats_df['match_id'] = match_metadata.get('matchId', 'N/A')
        total_matchstats_df['gameMode'] = match_info['gameMode']
        total_matchstats_df['gameDuration'] = match_info['gameDuration']
        total_matchstats_df['platform_id'] = match_info['platformId']

        print('Complete Match Data:')
        print(total_matchstats_df)

        append_to_database(total_matchstats_df, 'total_matchstats')
    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')



get_matches()
