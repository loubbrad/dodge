import torch
import pandas as pd
import sqlite3
from torch.utils.data import Dataset
from datetime import datetime
from progress.bar import Bar

# TODO:
# - Add assert statements and doc strings
# - Change so that we use summoner_id instead of puuid

class SQLConnection():
    def __init__(self, db_path: str, patches = 'all', regions = 'all'):
        assert regions == 'all' or type(regions) ==list, 'regions must be a list'
        assert patches == 'all' or type(patches) == list, 'patches must be a list'

        self.db_path = db_path
        self.patches = patches
        self.regions = regions
        
        # Init database connection
        self.conn = sqlite3.connect(db_path)
        self.table_name = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchone()[0]
        
    def get_df(self):
        return pd.read_sql_query(self._generate_query(), self.conn)
    
    def _generate_query(self):
        # Generate query for cols
        select_cols = 'match_id, blue_win, created_at, '
        for i in range(1, 10):
            select_cols += f"p{i}_puuid, p{i}_name, p{i}_champ, p{i}_team, p{i}_position, "
        select_cols += 'p10_puuid, p10_name, p10_champ, p10_team, p10_position'
    
        # Generate query for regions
        if self.regions == 'all':
            select_reg = self.table_name
        else:
            select_reg = f"(SELECT * FROM {self.table_name} WHERE"
            for reg in self.regions[:-1]:
                select_reg += f" region='{reg}' OR"
            select_reg += f" region='{self.regions[-1]}')"

        # Generate query for patches
        if self.patches == 'all':
            patches_condition = ''
        else:
            patches_condition = """ WHERE"""
            for patch in self.patches[:-1]:
                patches_condition += f" patch LIKE '{patch}%' OR"
            patches_condition += f" patch LIKE '{self.patches[-1]}%'"
            
        # Combine and return
        return 'SELECT ' + select_cols + ' FROM ' + select_reg + patches_condition


class MatchDataset(Dataset):
    def __init__(self, device, sql_conn: SQLConnection, seed=42, test_train_ratio=0.90, champ_only=True):
        self.device = device
        self.champ_only = champ_only

        print('Executing SQL...')
        raw_data = sql_conn.get_df()
        print('Building dicts...')
        self._init_dicts(raw_data)
        processed_data = self._preprocess(raw_data)

        # Shuffle data
        torch.manual_seed(seed)
        shuffle_ind = torch.randperm(processed_data.size()[0])
        processed_data = processed_data[shuffle_ind]
        
        # Test train split
        test_train_idx = round(test_train_ratio*processed_data.shape[0])
        self.train, self.dev = processed_data[:test_train_idx], processed_data[test_train_idx:]

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        """Returns (match, outcome, time) by id."""
        match = self.train[idx, 2:]
        outcome = self.train[idx, 0].to(torch.float)
        time = self.train[idx, 1]
        return match, outcome, time
        
    def get_dev(self): 
        """Returns entire dev set in correct format"""
        matches = self.dev[:, 2:]
        outcomes = self.dev[:, 0].to(torch.float)
        times = self.dev[:, 1]
        return matches, outcomes, times

    def get_train(self):
        """Returns entire train set in correct format"""
        matches = self.train[:, 2:]
        outcomes = self.train[:, 0].to(torch.float)
        times = self.train[:, 1]
        return matches, outcomes, times

    def _init_dicts(self, raw_data: pd.DataFrame):
        # Calculate list of champions present
        champs_present = []
        for i in range(1, 11):
            champs_present += list(raw_data[f'p{i}_champ'].unique())
        champs_present = list(set(champs_present))

        # champ_id (riot) <-> idx
        # 0 is a special padding character
        self.key_to_champid = {k+1:v for k, v in enumerate(champs_present)}
        self.champid_to_key = {v:k+1 for k, v in enumerate(champs_present)}
        self.num_champs = len(champs_present)+1

        # Only calculate if require player info (memory reasons)
        if self.champ_only == False:
            # Gather list of unique puuids
            players_present = []
            for i in range(1, 11):
                players_present += list(raw_data[f'p{i}_puuid'])
                players_present = list(set(players_present))
                
            self.num_players = len(players_present)
                
            # key <-> puuid
            self.key_to_puuid = {key: puuid for key, puuid in enumerate(players_present)}
            self.puuid_to_key = {puuid: key for key, puuid in enumerate(players_present)}
    
    def _preprocess(self, raw_data: pd.DataFrame):
        # Reorganise the rows by looping through them
        # This is very inefficient, should ideally be refactored

        POS_DICT = {'BLUE': {'TOP': 1, 'JUNGLE': 2, 'MID': 3, 'ADC': 4, 'SUPPORT': 5},
                    'RED': {'TOP': 6, 'JUNGLE': 7, 'MID': 8, 'ADC': 9, 'SUPPORT': 10}}
                          
        matches = []
        with Bar('Preprocessing matches...', max=raw_data.shape[0]) as bar:
            for _, raw_match in raw_data.iterrows():
                match = {}
                match['blue_win'] = raw_match['blue_win']
                match['created_at'] = int(datetime.strptime(raw_match['created_at'][:-6], "%Y-%m-%dT%H:%M:%S").timestamp()) # Convert to epoch
                for i in range(1, 11):
                    team_temp = raw_match[f'p{i}_team']
                    position_temp = raw_match[f'p{i}_position']
                    new_idx = POS_DICT[team_temp][position_temp]
                    
                    match[f'p{new_idx}_champ'] = self.champid_to_key[raw_match[f'p{i}_champ']]
                    if self.champ_only == False: # TODO: Refactor this so the code is cleaner
                        match[f'p{new_idx}_id'] = self.puuid_to_key[raw_match[f'p{i}_puuid']]
                
                matches.append(match)
                bar.next()
        
        # Column ordering
        ordered_cols = ['blue_win', 'created_at']
        for i in range(1, 11):
            # Only had p_id to DataFrame if specified by self.champ_only
            if self.champ_only == False:
                ordered_cols.append(f'p{i}_id')
            ordered_cols.append(f'p{i}_champ')

        # Format for return
        processed_data = pd.DataFrame.from_records(matches).astype(int)[ordered_cols]
        processed_data = torch.tensor(processed_data.values, dtype=torch.long).to(self.device)
        
        return processed_data


class PlayerHistory(Dataset):
    def __init__(self, match_dataset: MatchDataset, history_len: int): 
        self.history_len = history_len
        self.match_history = self._preprocess(match_dataset)
        
    def __len__(self):
        return self.match_history.shape[0]

    def __getitem__(self, idx):
        return self.match_history[idx]
        
    def _preprocess(self, match_dataset): 
        # Build dictionary of match_history
        match_history_dict = {}
        for i in range(match_dataset.num_players):
            match_history_dict[i] = []
        with Bar('Building match history...', max=len(match_dataset)) as bar:
            for idx in range(len(match_dataset)):
                match, outcome, time = match_dataset[idx]
                for i in range(0, 10, 2):
                    match_history_dict[match[i].item()].append([match[i+1].item(), outcome.item(), time.item()]) # p1-5
                for i in range(10, 20, 2):
                    match_history_dict[match[i].item()].append([match[i+1].item(), 1-outcome.item(), time.item()]) # p6-10
                bar.next()

        # Pad and sort
        for k, match_history in match_history_dict.items():
            # Pad if necessary
            match_history_len = len(match_history)
            if match_history_len < self.history_len:
                match_history += [[0, -1, 0]]*(self.history_len - match_history_len) # champ=0, outcome=-1, time=0
            # Sort according to time
            match_history.sort(key=lambda x: x[2], reverse=True) 
            # Cut off the end
            match_history_dict[k] = match_history[:self.history_len]

        # Process into tensor
        return torch.stack([torch.tensor(match_history_dict[i], dtype=torch.long) for i in range(len(match_history_dict))], dim=0).to(match_dataset.device)
        

def test():
    return

if __name__ == '__main__':
    test()
                        