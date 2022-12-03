import pandas as pd
import crawl
import process 

def main():

    # EXAMPLE USAGE:

    # Find player data for top 200 players:
    raw_player_data = crawl.load_players('euw1', 200, save_path = './data/raw/')

    # Find match history of all players in raw_player_data between 1667937600 and time.time():
    # 1667937600 = 20:00 GMT 08/11/2022 (UK date format).
    raw_match_data = crawl.load_matches('euw1', 1667433600, players = raw_player_data, save_path = './data/raw/', time_limit = 10)
    
    # Processing saving match and player data:
    proc_match_data = process.process_matches(match_data = raw_match_data, player_data = raw_player_data , save_path = './data/processed/')

if __name__ == '__main__':
    main()