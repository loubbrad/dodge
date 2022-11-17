import pandas as pd
import crawl
import process 

def main():

    # EXAMPLE USAGE:

    # Find player data for top 100 players:
    raw_player_data, raw_player_data_file_name = crawl.load_players('euw1', 1000, './data/raw/')

    # Find match history of all players in raw_player_data between 1667937600 and time.time():
    # 1667937600 = 20:00 GMT 08/11/2022 (UK date format).
    raw_match_data, raw_match_data_file_name = crawl.load_matches('euw1', 1667433600, players = raw_player_data, save_path = './data/raw/', time_limit = 24)
    raw_match_data_path  = './data/raw/' + raw_match_data_file_name
    
    # Processing player_data:
    proc_player_data, proc_player_data_file_name = process.process_player_data_from_matches(raw_match_data_path, './data/processed/')
    proc_player_data_path = './data/processed/' + proc_player_data_file_name
    
    # Processing match data:
    proc_match_data = process.process_match_data(raw_match_data_path, proc_player_data_path, './data/processed/', noise_rate = 0.1)

if __name__ == '__main__':
    main()