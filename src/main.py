import pandas as pd
import extract_lol_data as eld

def main():

    df = eld.load_players('na1', 1000, './data/raw/') 
    eld.load_matches('na1', 1665964800, players = df, save_path = './data/raw/', time_limit= 10)

if __name__ == '__main__':
    main()