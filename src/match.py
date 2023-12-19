import requests
import pandas as pd
import os
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
import time
import random

puuids = [
  '26AyL0tOH0kFRimTlUIUgQKYaSHzzPWgKfyUeG6anJmp4mQn7xalvsv2p-gfd69IG-8Xu0-iNbHE5A',
  '5nQ9C4WoUW1Sh77ycj25diDSchnuY5mvCT5ChfJyEhBTjRDFqN50qQE2DsAbRAHzo4FbTVf7jhyffw',
  '4kKOYUIOIVoWrbPr8JESr7l_JLizWOemuSySFgnDZk1qrdCKmINukax-732AAh6qG_WttH_Qi2ExLA',
  'gbCJkmT5muTMKCX1ZS6LmoRtOXfeLU7qJebVungHTp7VCk9Slf4_bxSbrVOtLxqNNReeStNEQi5PLA',
  'Ud2nyUv5nYT2JdcYNyla_MZmEUnLePzxa_1D6pTGiGbkoh0yA7JbBJM4ZI3yyCaezMcTF5XusepB3g'
]


riot_api_base_url = 'https://americas.api.riotgames.com/'
api_key = 'RGAPI-8ea7f2b5-052f-4a16-9b0b-f97786be472b'
headers = {'X-Riot-Token':api_key}
summoner_name = 'Kailinho'
database_uri = 'postgresql://kai:kkaakkaa@localhost:5432/lol_matchinfo'
engine = create_engine(database_uri)
participants_df = pd.DataFrame()
events_df = pd.DataFrame()
total_matchstats_df = pd.DataFrame()

def get_matches(puuid, requests_per_minute=50, cooldown_duration=30):
    global participants_df, events_df, total_matchstats_df
    try:
        match_list_url = f'{riot_api_base_url}lol/match/v5/matches/by-puuid/{puuid}/ids?start=1&count=50&api_key={api_key}'

        response = requests.get(match_list_url , headers=headers)
        match_id_list = response.json()

        print(f'Match IDs for PUUID {puuid}: {match_id_list}')

        for match_id in match_id_list:
           get_match_info(match_id)
           time.sleep(60 / requests_per_minute)

    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')

    time.sleep(cooldown_duration)

def get_more_players(initial_puuids, num_additional_players=8):
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



def collect_aggregated_data(frames, match_id):
    global participants_df, events_df

    participants_columns = ['match_id', 'participant_id', 'currentGold', 'magicDamageDone', 'physicalDamageDone',
                             'trueDamageDone', 'magicDamageTaken', 'physicalDamageTaken', 'trueDamageTaken', 'xp',
                             'timeEnemySpentControlled', 'totalGold', 'timestamp']

    events_columns = ['match_id', 'participant_id', 'kills', 'deaths', 'assists', 'dragon_kills', 'turret_plates',
                      'inhibitor_kills']

    participants_df = pd.DataFrame(columns=participants_columns)
    events_df = pd.DataFrame(columns=events_columns)
    events_data = {'match_id': {}, 'participant_id': {}, 'kills': {}, 'deaths': {}, 'assists': {}, 'timestamp': {},
                   'dragon_kills': {}, 'turret_plates': {}, 'inhibitor_kills': {}, }

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


            elif event['type'] == 'ELITE_MONSTER_KILL' and event['monsterType'] == 'DRAGON':
                team_id = event['killerTeamId']
                participant_id = event['killerId']
                dragon_kills = events_data['dragon_kills'].get(participant_id, 0) + 1
                events_data['dragon_kills'][participant_id] = dragon_kills
                if team_id == 200 and 6 <= participant_id <= 10:
                    events_data['dragon_kills'][participant_id] += 1
                elif team_id == 100 and 1 <= participant_id <= 5:
                    events_data['dragon_kills'][participant_id] += 1

            elif event['type'] == 'TURRET_PLATE_DESTROYED':
                killer_id = event['killerId']
                turret_plates_destroyed = events_data['turret_plates'].get(killer_id, 0) + 1
                events_data['turret_plates'][killer_id] = turret_plates_destroyed

            elif event['type'] == 'BUILDING_KILL' and event['buildingType'] == 'INHIBITOR_BUILDING':
                participant_id = event['killerId']
                inhibitor_kills = events_data['inhibitor_kills'].get(participant_id, 0) + 1
                events_data['inhibitor_kills'][participant_id] = inhibitor_kills

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

        # Check if 'metadata' key is present in match_data
        if 'metadata' not in match_data:
            print('Error: No "metadata" in match data')
            return

        match_metadata = match_data['metadata']
        match_info = match_data['info']

        if match_info.get('mapId', 0) != 11:
            print('Error: This match is not on Summoner\'s Rift')
            return

        match_timeline = match_timeline_data['info'].get('frames', [])
        
        # Collect aggregated data
        collect_aggregated_data(match_timeline,match_id)

        # Additional information for each participant from key_list
        key_list = [ 'pentaKills', 
                    'timeCCingOthers', 'totalMinionsKilled',
                    'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'turretKills', 'visionScore', 'win','championName']

        participants_data = []

        for participant_id in range(1, 11):
            participant_data = {'participant_id': participant_id}
            for key in key_list:
                # Extract additional participant information from key_list

                if key in match_info['participants'][participant_id - 1]:
                    participant_data[key] = match_info['participants'][participant_id - 1][key]

            # Add additional participant information to participants_data list
            participants_data.append(participant_data)

        total_matchstats_df = pd.DataFrame(columns=key_list)
        total_matchstats_df = pd.concat([total_matchstats_df, pd.DataFrame(participants_data)], ignore_index=True).fillna(0)
        total_matchstats_df[[ 'pentaKills', 
                 'timeCCingOthers', 'totalMinionsKilled',
                 'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'turretKills', 'visionScore', ]] = total_matchstats_df[['pentaKills', 
                 'timeCCingOthers', 'totalMinionsKilled',
                 'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'turretKills', 'visionScore', ]].astype(int)
        total_matchstats_df['match_id'] = match_metadata.get('matchId', 'N/A')
        total_matchstats_df['gameMode'] = match_info['gameMode']
        total_matchstats_df['gameDuration'] = match_info['gameDuration']
        total_matchstats_df['platform_id'] = match_info['platformId']

        print('Complete Match Data:')
        print(total_matchstats_df)

        append_to_database(total_matchstats_df, 'total_matchstats')
    except requests.exceptions.RequestException as error:
        print(f'Error: {error}')



additional_puuids = get_more_players(puuids)
all_puuids = puuids + additional_puuids

for puuid in all_puuids:
    get_matches(puuid)
