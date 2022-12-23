import pandas as pd
import torch

#TODO:
# - Add symmetric option for LoadMatchData
# - Add testing
# - Add option to reflect data

class MatchData():
    def __init__(self, load_path, **kwargs,):
        """Processes data into torch.tensor object.
        Args:
            load_path: path to .csv match_data file.
            data_format: 'champ', 'player' or None.
            device: 'cpu' or 'cuda or None.
            sequential: default = False (should be used for RNN)."""

        self.device = kwargs.get('device') 
        self.data_format = kwargs.get('data_format') 
        self.sequential = kwargs.get('sequential') 
        
        # Assert device is valid
        if self.device != None:
            assert self.device == 'cpu' or self.device == 'cuda', 'Invalid device.'
            self.device = self.device
        else:
            self.device = 'cpu'

        # Assert data_format is correct
        self.data_format = self.data_format
        if self.data_format != None:
            assert self.data_format == 'champ' or self.data_format == 'player', (
                'invalid data_format')
        
        try:
            raw_data = pd.read_csv(load_path)
        except Exception as e:
            print('Failed to load .csv')
            raise(e)
        
        try:
            proc_data = self._process_df(raw_data)
        except Exception as e:
            print('Failed to process data.')
            raise(e)
            
        # Data in tensor form
        self.X, self.Y = self._to_tensor(proc_data)

        if self.data_format == 'champ':
            self._champ_only()
        elif self.data_format == 'player':
            self._player_only()
        
        # Sequential
        if self.sequential == True:
            self._sequential()
            
        # Initialise test-train split and batch info to None
        self.split_size = None
        self.seed = None
        self.X_tr, self.Y_tr = self.X, self.Y
        self.X_te, self.Y_te = None, None
        self.batches = None

            
    def _process_df(self, raw_data):
        """Processes data and loads champ, player dictionaries."""

        # Drop unnecessary rows/cols
        col_list = ['match_id']
        col_list += ['p{j}_key'.format(j=i) for i in range(1,11)]
        col_list += ['p{j}_champId'.format(j=i) for i in range(1,11)]
        data = raw_data.drop(columns = col_list)

        # Create player_name dictionary
        pres_players = ['<U>']
        for i in range(1,11):
            pres_players += list(data[f'p{i}_name'].unique())
        pres_players = list(set(pres_players))

        self.num_players = len(pres_players)
        self.key_to_player = {k:v for k, v in enumerate(pres_players)}
        self.player_to_key = {k:v for v, k in enumerate(pres_players)}

        # Create champ_name dictionary
        pres_champs = []
        for i in range(1, 11):
            pres_champs += list(data[f'p{i}_champName'].unique())
        pres_champs = list(set(pres_champs))

        self.num_champs = len(pres_champs)
        self.key_to_champ = {k:v for k, v in enumerate(pres_champs)}
        self.champ_to_key = {k:v for v, k in enumerate(pres_champs)}
        
        # Format data according to champ and player dictionary
        for i in range(1, 11):
            data[f'p{i}_champName'] = data[f'p{i}_champName'].map(self.champ_to_key)
            data = data.rename(columns = {f'p{i}_champName': f'p{i}_champKey'})
            data[f'p{i}_name'] = data[f'p{i}_name'].map(self.player_to_key)
            data = data.rename(columns = {f'p{i}_name': f'p{i}_key'})
            
        return data
        

    def _to_tensor(self, data):
        """Converts processed data_frame to torch.tensor objects with correct dtypes."""
        
        # Inputs dtype='long' so we can use for embedding. Labels dtype='float' so we can use BCE loss.
        inputs = torch.tensor(data.drop(columns = ['winning_team']).values, dtype = torch.long).to(self.device)
        labels = torch.tensor(data.loc[:, ['winning_team']].values - 1, dtype = torch.float).to(self.device)
        
        return inputs, labels
    

    def _sequential(self):
        """Orders X as p1, p5, p2, p6,... , p5, p10 (or c respectively)."""
        assert self.data_format != None, 'Not yet supported for player&champ together.' # Add this feature
        
        ind = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        self.X = self.X[:, ind]


    def _champ_only(self):
        """Remove duplicated games and player info."""
        self.X = self.X[::2, 1::2]
        self.Y = self.Y[::2]


    def _player_only(self):
        """Remove champ info."""
        self.X = self.X[:, ::2]
        # Y unchanged
        
    
    def test_train_split(self, split_size:float, seed:int=42):
        """Splits data according to seed and self.data_format.
        Args:
            split_size: Proportion of data to include in train set.
            seed: Random seed (default 42).
        Returns:
            Tuples of the form (X_tr, Y_tr), (X_te, Y_te)."""
        
        # Check if arguments have changed since last call
        if self.seed == seed and self.split_size == split_size:
            return (self.X_tr, self.Y_tr), (self.X_te, self.Y_te)
            
        # Only run if parameters have changed (or are None)
        self.spit_size = split_size
        self.seed = seed
        torch.manual_seed(seed)
        if self.data_format == None: # i.e. both 'champ' and 'player'
            # Important - keeps training and testing separate
            X_temp = self.X.reshape(-1, 2, 20)
            Y_temp = self.Y.reshape(-1, 2) 
            
            # Randomly permute Xtemp, Ytemp
            rand_perm = torch.randperm(X_temp.shape[0])
            ind = round(split_size * X_temp.shape[0])
            X_temp, Y_temp = X_temp[rand_perm], Y_temp[rand_perm]
            X_tr, X_te = X_temp[:ind], X_temp[ind:]
            Y_tr, Y_te = Y_temp[:ind], Y_temp[ind:]
            
            # Reshape, set class attributes and return
            self.X_tr, self.Y_tr = X_tr.reshape(-1, 20), Y_tr.reshape(-1, 1)
            self.X_te, self.Y_te = X_te.reshape(-1, 20), Y_te.reshape(-1, 1)

            return (self.X_tr, self.Y_tr), (self.X_te, self.Y_te)

        elif self.data_format == 'champ' or self.data_format == 'player':
            X_temp, Y_temp = self.X.clone(), self.Y.clone()

            # Randomly permute Xtemp, Ytemp
            rand_perm = torch.randperm(X_temp.shape[0])
            ind = round(split_size * X_temp.shape[0])
            X_temp, Y_temp = X_temp[rand_perm], Y_temp[rand_perm]
            X_tr, X_te = X_temp[:ind], X_temp[ind:]
            Y_tr, Y_te = Y_temp[:ind], Y_temp[ind:]

            # Reshape, set class attributes and return
            self.X_tr, self.Y_tr = X_tr, Y_tr
            self.X_te, self.Y_te = X_te, Y_te

            return (self.X_tr, self.Y_tr), (self.X_te, self.Y_te)
        
        else:
            raise Exception('Invalid data_format attribute.')


    def create_mini_batches(self, batch_size:int):
        """Creates mini-batches out of Xtr and Ytr.
        Args:
            batch_size: Size of the batches
        Returns:
            list (of tuples) of batches [(X_tr_batch, Y_tr_batch)]."""
        
        # Testing if test-train split has occurred
        if self.X_te == None:
            print('WARNING: It is advised to execute a test-train split before creating mini-batches.')
            
        # Computing batches
        train_set_size = self.X_tr.shape[0]
        num_batches = train_set_size // batch_size
        assert num_batches >= 0, 'Batch size cannot exceed the size of the training data.'

        X_tr_b_list = [self.X_tr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        X_tr_b_list.append(self.X_tr[num_batches*batch_size:]) # Last batch may be incomplete
        Y_tr_b_list = [self.Y_tr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        Y_tr_b_list.append(self.Y_tr[num_batches*batch_size:]) # Last batch may be incomplete

        self.batches = [(X_tr_b, Y_tr_b) for X_tr_b, Y_tr_b in zip(X_tr_b_list, Y_tr_b_list)] 
        
        return self.batches