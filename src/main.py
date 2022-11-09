import pandas as pd
import extract_lol_data as eld
import process_lol_data as pld

def main():

    # EXAMPLE USAGE:

    # Find player data for top 100 players:
    raw_player_data, raw_player_data_file_name = eld.load_players('na1', 50, './data/raw/')

    # Find match history of all players in raw_player_data between 1667937600 and time.time():
    # 1667937600 = 20:00 GMT 08/11/2022 (UK date format).
    raw_match_data, raw_match_data_file_name = eld.load_matches('na1', 1667937600, players = raw_player_data, save_path = './data/raw/', time_limit = 0.2)
    raw_match_data_path  = './data/raw/' + raw_match_data_file_name
    
    # Processing player_data:
    proc_player_data, proc_player_data_file_name = pld.process_player_data_from_matches(raw_match_data_path, './data/processed/')
    proc_player_data_path = './data/processed/' + proc_player_data_file_name
    
    # Processing match data:
    proc_match_data = pld.process_match_data(raw_match_data_path, proc_player_data_path, './data/processed/', noise_rate = 0.1)

if __name__ == '__main__':
    main()