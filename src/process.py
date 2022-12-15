import pandas as pd
import numpy as np
import os
from riotwatcher import LolWatcher, ApiError

RIOT_API_KEY = os.environ.get("RIOT_API_KEY")

#TODO:
# - Add patch data (n for how many patches ago the match happened)
# - Add .start_date and .end_date attributes to dataframe.
# - Include metadata reading when reading csv files.
# - Move loading csv into it's own file (process_matches getting too long).
# - Specify dtypes of dataFrames (getting dtypeWarning at runtime).
# - Replace .format() with fstrings for readability.
# - Make sure code is PEP8 compliant. 

def process_matches (match_load_path: str | None = None, player_load_path: str | None = None,
                     match_data: pd.DataFrame | None = None, player_data: pd.DataFrame | None = None,
                     save_path: str | None = None):

    """
    Args:
        match_load_path: Path to raw_match_data.
        player_load_path: Path to raw_player_data.
        match_data: raw_match_data as DataFrame object.
        player_data: raw_player_data as DataFrame object.
        save_path: Path to save processed_match_data and processed_player_data.
    Returns:
        processed_match_data: Processed match data in DataFrame format. processed_match_data.file_name records filename if saved.
        processed_player_data: Processed player data in DataFrame format. processed_player_data.file_name records filename if saved.
    """

    # Loading player data into DataFrame.
    if (player_data is None) and (player_load_path is None):
        raise ValueError('Specify at least one of player_data or player_load_data')
    elif player_data is None:
        try:
            player_data = pd.read_csv(player_load_path)
        except FileNotFoundError as e:
            print('Invalid load path')
            raise
    if (match_data is None) and (match_load_path is None):
            raise ValueError('Specify at least one of match_data or match_load_path')
    elif match_data is None:
        try:
            match_data = pd.read_csv(match_load_path)
            match_data_file_name = os.path.split(match_load_path)[1]
        except FileNotFoundError as e:
            print('Invalid load path')
            raise
    else:
        match_data_file_name = match_data.file_name
    
    # Check for invalid API key.
    try:
        watcher = LolWatcher(RIOT_API_KEY) 
        latest_version = watcher.data_dragon.versions_for_region('na1')['n']['champion']
        static_champ_list = watcher.data_dragon.champions(latest_version, False, 'en_US')
    except ApiError as e:
        if e.response.status_code == 401:
            print('Invalid API key.')
            raise
        else:
            raise

    champ_dict = {int(static_champ_list['data'][champ]['key']):
              static_champ_list['data'][champ]['name'] for champ in static_champ_list['data']}
    
    # Processing player data.
    player_data = player_data.drop(columns = ['puuid', 'summonerId', 'accountId', 'leaguePoints'])
    player_data = player_data.rename(columns = {'summonerName': 'name'})
    player_data = pd.concat([pd.DataFrame([['<U>']], columns = ['name']), player_data], ignore_index = True)
    player_data['key'] = player_data.index

    # Adding column displaying the winning team and then deleting p{i}_win columns.
    match_data['winning_team'] = match_data.apply( lambda row: 2 - int(row.p1_win), axis = 1)
    
    for i in range(1, 11):
        
        # Remove unnecessary columns for player i.
        match_data = match_data.drop(columns = ['p{p_num}_puuid'.format(p_num = i),
                                                'p{p_num}_summonerId'.format(p_num = i),
                                                'p{p_num}_win'.format(p_num = i)])
        
        # Delete replace all players not present in player_data with '<U>'
        match_data[f'p{i}_name'] = match_data[f'p{i}_name'].apply(lambda name: name if name in player_data['name'].values else '<U>')
        
        # Add key column of keys for player i.
        match_data = match_data.join(player_data.set_index('name'), on = 'p{p_num}_name'.format(p_num = i))
        match_data = match_data.rename(columns = {'key': 'p{p_num}_key'.format(p_num = i )})

        # Adding column for name of champion of player i.
        match_data['p{p_num}_champName'.format(p_num = i)] = match_data['p{p_num}_champId'.format(p_num = i)].map(champ_dict)
        
    # Duplicate rows for team 1 and team 2, delete summoner information for corresponding team.
    match_data = pd.concat([match_data]*2).sort_index().reset_index(drop = True)
    match_data.loc[::2, ['p1_name', 'p2_name', 'p3_name', 'p4_name', 'p5_name']] = '<U>'
    match_data.loc[::2, ['p1_key', 'p2_key', 'p3_key', 'p4_key', 'p5_key']] = '0'
    match_data.loc[::2, ['match_id']] = match_data.loc[::2, ['match_id']] + '_1'
    match_data.loc[1::2, ['p6_name', 'p7_name', 'p8_name', 'p9_name', 'p10_name']] = '<U>'
    match_data.loc[1::2, ['p6_key', 'p7_key', 'p8_key', 'p9_key', 'p10_key']] = '0'
    match_data.loc[1::2, ['match_id']] = match_data.loc[1::2, ['match_id']] + '_2'
    
    # Reordering columns.
    ordered_cols = ['match_id', 'winning_team']
    for i in range(1, 11):
        ordered_cols.append('p{p_num}_name'.format(p_num = i))
        ordered_cols.append('p{p_num}_key'.format(p_num = i))
        ordered_cols.append('p{p_num}_champId'.format(p_num = i))
        ordered_cols.append('p{p_num}_champName'.format(p_num = i))
        
    match_data_processed = match_data[ordered_cols]
    player_data_processed = player_data[['name', 'key']]
    
    # Saving data.
    match_data_processed.file_name = match_data_file_name.replace('raw_match_data', 'processed_match_data')
    player_data_processed.file_name = match_data_file_name.replace('raw_match_data', 'processed_player_data')
    if not (save_path is None):
        try:
            match_data_processed.to_csv(save_path + match_data_processed.file_name, index = False)
            player_data_processed.to_csv(save_path + player_data_processed.file_name, index = False)
        except Exception as e:
            print(e)
            print('Failed to save processed match data to .csv file.')
        else:
            return match_data_processed, match_data_file_name.replace('raw_match_data', 'processed_match_data')

    return match_data_processed, player_data_processed