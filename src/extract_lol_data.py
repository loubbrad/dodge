import os
import time
import pandas as pd
from riotwatcher import LolWatcher, ApiError

RIOT_API_KEY = os.environ.get("RIOT_API_KEY")

# TODO:
# - Complete testing
# - Rename variables for clarity.

def load_players(reg: str, player_count: int = 1000, save_path: str | None = None):
    
    """
    Args:
        reg: Region. 
        player_count: Number of players to gather data on, sorted by LP ladder.
        save_path: Path to save player data if save == True.

    Returns:
        player_data: Player data in DataFrame format.
        file_name: Name of saved file (optional).
    """
    
    # Assert Args have correct format.
    assert reg in ['na1', 'euw1', 'eun1' 'kr'], 'Invalid region.'
    assert player_count > 0, 'Invalid player count (must be >0).'

    # Catch invalid API key.
    try:
        watcher = LolWatcher(RIOT_API_KEY)
        watcher.lol_status_v4.platform_data(region = reg)
    except ApiError as e:
        if e.response.status_code == 401:
            print('Invalid API key.')
            raise
        else:
            raise

    # Gather player information into DataFrame format.

    chal_players = watcher.league.challenger_by_queue(region = reg, queue = 'RANKED_SOLO_5x5')
    gm_players = watcher.league.grandmaster_by_queue(region = reg, queue = 'RANKED_SOLO_5x5')
    m_players = watcher.league.masters_by_queue(region = reg, queue = 'RANKED_SOLO_5x5')

    j = 0
    in_order_players = []

    for summoner, i in zip(chal_players['entries'] + gm_players['entries'] + m_players['entries'],
                           range(1, player_count + 1)):

        try:
            summoner_info = watcher.summoner.by_id(encrypted_summoner_id = summoner['summonerId'], region = reg)
        except ApiError as e:
            print('An exception as occurred at when processing ' + str(summoner_info['summonerName']) + ':')
            print(e)
            j += j
        else:
            in_order_players.append((summoner_info['puuid'], summoner_info['accountId'],
                                     summoner['summonerId'], summoner['summonerName'],
                                     summoner['leaguePoints']))

        if i % 100 == 0: 
            print( i, 'players processed.')

    print(j, 'failures.')
    in_order_players = sorted(in_order_players, key = lambda kv: kv[4], reverse = True)

    df = pd.DataFrame(in_order_players, columns = ['puuid', 'accountId', 'summonerId', 'summonerName', 'leaguePoints'])
    df = df.sort_values('leaguePoints', ascending = False)
    
    # Saving player data.
    if not (save_path is None) :
        curr_date = round(time.time())
        try:
            df.to_csv( (save_path + '{name}.csv').format(name = 'raw_player_data_' + reg + '_' + str(curr_date)) , index = False)
        except Exception as e:
            print('Failed to save player data to .csv file.')
            raise
        else:
            return df, '{name}.csv'.format(name = 'raw_player_data_' + reg + '_' + str(curr_date))
        
    return df

def load_matches(reg: str, date: int, players: pd.DataFrame | None = None,
                 load_path: str | None = None, save_path: str | None = None,
                 time_limit: float | None = None):
    
    """
    Args:
        reg: Region.
        date: Epoch (minute) timestamp of how far to look back for matches.
        players: Player data in DataFrame format with column labelled 'puuid'.
        load_path: Path to .csv file where first column corresponds to player puuids.
        save_path: Path to save match data.
        time_limit: Time in hours as an upper limit for the script to run.

    Returns:
        match_data: Match data in DataFrame format.
        file_name: Name of saved file (optional).
    """ 
    
    # Assert Args have correct format.
    assert reg in ['na1', 'euw1', 'eun1' 'kr'], 'Invalid region'
    assert time_limit > 0, 'Invalid time limit (must be >0).'
    assert date < time.time(), 'Invalid date (must be < current Epoch minute time)'

    # Loading player data into DataFrame.
    if (players is None) and (load_path is None):
        raise ValueError('Specify at least one of players or load_data')
    elif players is None:
        try:
            player_puuids = pd.read_csv(load_path)
            player_puuids = player_puuids.loc[:,'puuid']
        except KeyError:
            print("Data must have column with key 'puuid'.")
            raise
    else:
        try:
            player_puuids = players.loc[:,'puuid']
        except KeyError:
            print("Data must have column with key 'puuid'.")
            raise
    
    # Catch invalid API key.
    try:
        watcher = LolWatcher(RIOT_API_KEY)
        watcher.lol_status_v4.platform_data(region = reg)
    except ApiError as e:
        if e.response.status_code == 401:
            print('Invalid API key.')
            raise
        else:
            raise
        
    # Finding matches 

    searched_matches = []
    match_data = {}  
    curr_time = round(time.time() - 1)
    prev = 0
    i = 0

    for puuid in player_puuids:

        player_match_history, searched_matches = load_matches_from_player(watcher, puuid, date, curr_time, reg, searched_matches)
        match_data = match_data | player_match_history

        if (len(searched_matches)) - prev > 100:
            print('Processed ' + str(len(searched_matches)) + ' matches.')
            prev = len(searched_matches) 

        i += 1
        
        if (time.time() - curr_time) >= time_limit * (60**2): 
            print('Time limit of ' + str(time_limit) + ' hours exceeded.')
            print(i, 'out of', player_puuids.size, "player's match history processed.")
            break 
    
    df = pd.DataFrame.from_dict(match_data, orient = 'index')
    df.index.name = 'match_id'
    
    # Saving matches
    if not (save_path is None):
        try:
            df.to_csv((save_path + '{name}.csv').format(name = 'raw_match_data_' + reg + '_' + str(date) + '_' + str(curr_time)))
        except Exception as e:
            print('Failed to save match data to .csv file.')
            raise
        else:
            return df, '{name}.csv'.format(name = 'raw_match_data_' + reg + '_' + str(date) + '_' + str(curr_time))
    
    return df

def load_matches_from_player(watcher: LolWatcher, puuid: str, start_date: int, end_date: int, reg: str, searched_matches = []):
    
    """
    Args:
        watcher: LolWatcher object.
        puuid: puuid of player to search.
        start_date: Epoch (minute) timestamp of how far back to look for matches.
        end_date: Epoch (minute) timestamp of how far forward to look for matches (end_date > start_date).
        reg: Region.
        searched_matches: List of matches already searched.

    Returns:
        player_match_data: Dictionary of the form {match_id: {attribute: value}}
        searched_matches: Updated list of matched already searched.
    """

    try:
        match_list = watcher.match.matchlist_by_puuid(puuid = puuid, region = reg, start_time = start_date, end_time = end_date, count = 100)
    except ApiError as e:
        print('There was an issue finding the matches of player' + str(puuid) + ':')
        print(e)
        return  
    
    player_match_data = {}

    for match_id in match_list:
        
        if match_id not in searched_matches:
            
            try:
                match_info = watcher.match.by_id(region = reg, match_id = match_id)
            except ApiError as e:
                print('There was an issue finding the match information for match_id =' + str(match_id) + ':')
                print(e)
                break

            serialised_match_info = {}
            i = 0
                                    
            for participant in match_info['info']['participants']:
                
                i += 1
                
                serialised_match_info['p{p_num}'.format(p_num = i) + '_name'] = participant['summonerName']
                serialised_match_info['p{p_num}'.format(p_num = i) + '_puuid'] = participant['puuid']
                serialised_match_info['p{p_num}'.format(p_num = i) + '_summonerId'] = participant['summonerId']
                serialised_match_info['p{p_num}'.format(p_num = i) + '_champId'] = participant['championId']
                serialised_match_info['p{p_num}'.format(p_num = i) + '_win'] = participant['win']

            player_match_data[match_id] = serialised_match_info
            searched_matches.append(match_id)
        
    return player_match_data, searched_matches