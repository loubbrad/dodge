import torch
from torch import nn as nn
from datasets import PlayerHistory
from nn_modules.modules import HistoryEncoder, TeamCompEncoder
from nn_modules.utils import PositionalEncoding

class CombinedTransformerModel(nn.Module):
    """Include here"""
    def __init__(self, player_history: PlayerHistory, num_champs: int,
                 h_embed_dim: int, h_num_heads: int, tc_embed_dim: int,
                 tc_num_heads: int, d_prob=0.1):

        super(CombinedTransformerModel, self).__init__()
        self.player_history = player_history
        self.h_embed_dim = h_embed_dim
        self.tc_embed_dim = tc_embed_dim
        self.h_num_heads = h_num_heads
        self.tc_num_heads = tc_num_heads
        self.name = 'combined'

        # History encoder, position-wise ff layer, add + layer norm
        self.history_encoder = HistoryEncoder(num_champs, h_embed_dim, h_num_heads, d_prob)
        self.h_linear_ff = nn.Linear(h_embed_dim, h_embed_dim)
        self.h_relu_ff = nn.ReLU()
        self.h_dropout_ff = nn.Dropout(d_prob)
        self.h_layer_norm = nn.LayerNorm(h_embed_dim, eps=1e-5) 
    
        # Team comp encoder ((position-wise) ff, add + layer norm included in TeamCompEncoder)
        self.team_comp_encoder = TeamCompEncoder(num_champs, tc_embed_dim, tc_embed_dim, d_prob)

        # FF network
        layer_size = 10*h_embed_dim + 10*tc_embed_dim
        self.flatten = nn.Flatten()
        self.linear_ff = nn.Linear(layer_size, layer_size)
        self.relu_ff = nn.ReLU()
        self.dropout_ff = nn.Dropout(d_prob)

        # Output
        self.linear_out = nn.Linear(layer_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, match, time):
        """
        Args:
            match: Tensor of shape (#batches, 20).
            times: Tensor of shape (#batches, 1).
        Returns:
            probs: Tensor of shape (#batches, 1).""" 
    
        # Encode history
        history = self._history_encoder(*self._history_format_data(match, time)) # shape (#batches, 10, h_embed_dim)
        # Encode team comp
        team_comp = self._team_comp_encoder(match) # shape (#batches, 10, tc_embed_dim)
        # Flatten and Concatenate
        combined = torch.cat((self.flatten(history), self.flatten(team_comp)), dim=1) # shape (#batches, 10*h_embed_dim + 10*tc_embed_dim)
        # MLP
        mlp = self.dropout_ff(self.relu_ff(self.linear_ff(combined)))
        # Output
        probs = self.sigmoid(self.linear_out(mlp))
    
        return probs
    
    def _history_format_data(self, match, time):
        """Formats matches and times so that they can be fed into a HistoryEncoder module.
        Args:
            matches: Tensor of shape (#batches, 20)
            times: Tensor of shape (#batches, 1)
        Returns:
            past_matches: Tensor of shape (#batches*10, history_len, 3).
            curr_match: Tensor of shape (#batches*10, 2)."""

        player_ids = match[:, 0::2]
        past_matches = self.player_history[player_ids] # shape (#batches, 10, history_len, 3)
        curr_match_champs = match[:, 1::2].view(-1, 10, 1) # shape (#batches, 10, 1)
        curr_match_times = time.view(-1, 1, 1).expand(-1, 10, 1).clone() # shape (#batches, 10, 1)
        curr_match = torch.cat((curr_match_champs, curr_match_times), dim=2) # shape (#batches, 10, 2) 
        
        return past_matches.flatten(0, 1), curr_match.flatten(0, 1) # Flatten dim=0,1
        
    def _history_encoder(self, past_matches, curr_match):
        """ Attention -> (position-wise) ff layer -> add + norm.
        Args:
            past_matches: Tensor of shape (#batches*10, history_len, 3).
            curr_match: Tensor of shape (#batches*10, 2).
        Returns:
            att: Tensor of shape (#batches, 10, h_embed_dim).""" 

        # Attention
        att = self.history_encoder(past_matches, curr_match).reshape(-1, 10, self.h_embed_dim) # shape (#batches. 10, h_embed_dim)
        # (position-wise) FF layer
        ff_layer = self.h_dropout_ff(self.h_relu_ff(self.h_linear_ff(att)))
        # Add + Norm
        normalised = self.h_layer_norm(att + ff_layer)
        
        return normalised

    def _team_comp_encoder(self, match):
        """ Attention -> (position-wise) ff layer -> add + norm.
        Args:
            match: Tensor of shape (#batches, 20).
        Returns:
            match_enc: Tensor of shape (#batches, 10, tc_embed_dim)."""

        champ_ids = match[:, 1::2]

        return self.team_comp_encoder(champ_ids)


class HistoryTransformerModel(nn.Module):
    """Given a PlayerHistory object, models the outcome of a (match, time) pair
    using the player's match history (masked to exclude the current match)."""
    def __init__(self, player_history: PlayerHistory, num_champs: int,
                 embed_dim: int, num_heads: int, d_prob = 0.1):
        super(HistoryTransformerModel, self).__init__()
        self.player_history = player_history
        self.embed_dim = embed_dim
        self.name = 'player_history'
        layer_size = 10*embed_dim 
        
        # Encoder layer
        self.history_encoder = HistoryEncoder(num_champs, embed_dim, num_heads, d_prob)

        # FF network
        self.flatten = nn.Flatten()
        self.linear_ff = nn.Linear(layer_size, layer_size)
        self.relu_ff = nn.ReLU()
        self.dropout_ff = nn.Dropout(d_prob)
        
        # Add + Norm layer
        self.layer_norm = nn.LayerNorm(layer_size) # Correct?

        # Output
        self.linear_out = nn.Linear(layer_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, match, time):
        """
        Args:
            match: Tensor of shape (#batches, 20)
            times: Tensor of shape (#batches, 1)
        Returns:
            probs: Tensor of shape (#batches, 1)""" 
            
        past_matches, curr_match = self._format_data(match, time) # 
        encoded = self.history_encoder(past_matches, curr_match).reshape(-1, 10, self.embed_dim) # Embed and calculate attention
        flattened = self.flatten(encoded) # Flatten 
        add = flattened + self._ff_layer(flattened) # Residual connection
        normalised = self.layer_norm(add) # Layer norm
        probs = self.sigmoid(self.linear_out(normalised))
        
        return probs
        
    def _format_data(self, match, time): 
        """Formats matches and times so that they can be fed into a HistoryEncoder module.
        Args:
            matches: Tensor of shape (#batches, 20)
            times: Tensor of shape (#batches, 1)
        Returns:
            past_matches: Tensor of shape (#batches*10, history_len, 3).
            curr_match: Tensor of shape (#batches*10, 2)."""

        player_ids = match[:, 0::2]
        player_histories = self.player_history[player_ids] # shape (#batches, 10, history_len, 3)
        curr_match_champs = match[:, 1::2].view(-1, 10, 1) # shape (#batches, 10, 1)
        curr_match_times = time.view(-1, 1, 1).expand(-1, 10, 1).clone() # shape (#batches, 10, 1)
        curr_match = torch.cat((curr_match_champs, curr_match_times), dim=2) # shape (#batches, 10, 2) 
        
        return player_histories.flatten(0, 1), curr_match.flatten(0, 1) # Flatten dim=0,1

    def _ff_layer(self, x): 
        y = self.dropout_ff(self.relu_ff(self.linear_ff(x)))

        return y
        

class TeamCompTransformerModel(nn.Module):
    """Models the outcome of a match of the form [p1_champ, p2_champ, ..., p10_champ] using
    a multi-layer Transformer Encoder architecture."""
    def __init__(self, num_champs: int, embed_dim: int, num_heads: int, d_prob = 0.1):
        super(TeamCompTransformerModel, self).__init__()
        self.name = 'team_comp'

        # Transformer Encoder
        self.embed = nn.Embedding(num_champs, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                  dropout=d_prob, dim_feedforward=embed_dim,
                                                  batch_first=True)
        self.team_comp_encoder = nn.TransformerEncoder(self.encode_layer, 2)

        # Output
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(embed_dim*10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, match):
        """
        Args:
            match: Tensor of shape (#batches, 10).
        Returns:
            probs: Tensor of shape (#batches, 1)."""
        
        match_emb = self.embed(match)
        encoded = self.team_comp_encoder(match_emb)
        flattened = self.flatten(encoded)
        probs = self.sigmoid(self.linear(flattened))
        
        return probs


def test():
    return

if __name__ == '__main__':
    test()