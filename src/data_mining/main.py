import os
import pandas as pd
from riotwatcher import LolWatcher, ApiError

# current api key: RGAPI-a50279df-996b-416d-adbc-eaaac13f8840
RIOT_API_KEY = os.environ.get("RIOT_API_KEY")

# TODO:
# - Complete load_matches method.
# - Reformat the file for external use.

def main():
    
    data_path = 'data/'
    reg = 'na1'
    
    # catch invalid API key
    try:
        watcher = LolWatcher(RIOT_API_KEY)
        watcher.lol_status_v4.platform_data(region = reg)
    except ApiError as e:
        if e.response.status_code == 401:
            print('Invalid API key.')
        else:
            raise
    else:
        del watcher
    
    player_data = load_players(data_path, reg, False)
    match_data = load_matches(player_data)

def load_players(data_path, reg, save = False):
    
    """
    Args:
        data_path: Path to save player data if save == True.
        region: Region of players.

    Returns:
        Player data in dataFrame format.
    """

    watcher = LolWatcher(RIOT_API_KEY)
    chal_players = watcher.league.challenger_by_queue(region = reg, queue = 'RANKED_SOLO_5x5')
    gm_players = watcher.league.grandmaster_by_queue(region = reg, queue = 'RANKED_SOLO_5x5')

    i = j = 0
    in_order_players = []

    for summoner in chal_players['entries'] + gm_players['entries']:

        try:
            summoner_info = watcher.summoner.by_id(encrypted_summoner_id = summoner['summonerId'], region = reg)
        except ApiError as e:
            print('An exception as occurred at when processing ' + str(summoner_info['summonerName']) + ':')
            print(e)
            j += j
        else:
            in_order_players.append( (summoner_info['puuid'], summoner_info['accountId'], summoner['summonerId'], summoner['summonerName'], summoner['leaguePoints']) )

        i += 1
        if i % 50 == 0: 
            print( i, 'players processed.')

    print(j, 'failures.')
    in_order_players = sorted(in_order_players, key = lambda kv: kv[4], reverse = True)

    df = pd.DataFrame(in_order_players, columns = ['puuid', 'accountId', 'summonerId', 'summonerName', 'leaguePoints'])
    df = df.sort_values('leaguePoints', ascending = False)
    if save == 'y':
        df.to_csv( (data_path + '{name}.csv').format(name = 'players') , index = False)

    return df

def load_matches():
    return 0
    
if __name__ == '__main__':
    main()