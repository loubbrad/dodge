import torch
from torch import nn as nn
from .utils import AttentionBlock, PositionalEncoding


class TeamCompEncoder(nn.Module):
    """Given a current match of the form [p1_champ, p2_champ, ..., p10_champ] embed and
    perform self attention -> (position-wise) feed forward layer -> add + layer norm."""
    def __init__(self, num_champs: int, embed_dim: int, num_heads: int, d_prob = 0.1):
        super(TeamCompEncoder, self).__init__()
        
        self.embed = nn.Embedding(num_champs, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                  dropout=d_prob, dim_feedforward=embed_dim,
                                                  batch_first=True)
        
    def forward(self, x):
        embedded = self.embed(x)
        pos_encoded = self.pos_encoder(embedded)
        encoded = self.encode_layer(pos_encoded)
        
        return encoded


class HistoryEncoder(nn.Module):
    """Given a current match of the form [champ_id, time] and a match history of the form
    [[champ_id, outcome, time], ...], embed and perform attention on the match history (to
    encode it with respect to the current match). Note this module only returns attention
    and does not perform (position-wise) feed forward and add + layer norm like TeamCompEncoder."""
    def __init__(self, num_champs: int, embed_dim: int, num_heads: int, d_prob = 0.1): # REMEBER to add back match_dataset
        super(HistoryEncoder, self ).__init__()

        self.embed_dim = embed_dim
        self.embed = nn.Embedding(num_champs, embed_dim, padding_idx=0)
        self.win_transform = nn.Linear(embed_dim, embed_dim, bias=False) # False required?
        self.loss_transform = nn.Linear(embed_dim, embed_dim, bias=False) # False required?

        self.att = AttentionBlock(embed_dim, num_heads)
        self.dropout = nn.Dropout(d_prob)
    
    def forward(self, past_matches, curr_match):
        """
        Args:
            past_matches: Tensor of shape (#batches, history_len, 3).
            curr_match: Tensor of shape (#batches, 2).
        Returns:
            att: (#batches, embed_dim)."""
   
        past_matches_emb, curr_match_emb = self._embed(past_matches, curr_match) 
        mask = self._calc_mask(past_matches, curr_match)
        att = self._att_layer(past_matches_emb, curr_match_emb, mask)

        return att
    
    def _embed(self, past_matches, curr_match): 
        """Embeds past_matches and curr_match.
        Args:
            past_matches: Tensor of shape (#batches, history_len, 3). 
            curr_match: Tensor of shape (#batches, 2).
        Returns:
            past_matches_emb: Tensor of shape (#batches, history_len, embed_dim).
            curr_match_emb: Tensor of shape (#batches, embed_dim)."""
        
        curr_match_emb = self.embed(curr_match[:, 0]) 
        past_champs, past_outcomes = past_matches[:,:,0], past_matches[:,:,1]
        past_outcomes_squ = torch.unsqueeze(past_outcomes, -1).expand(-1, -1, self.embed_dim)
        past_emb = self.embed(past_champs)
        past_emb_trans = self.win_transform(torch.where(past_outcomes_squ == 1, past_emb, 0)) + \
            self.loss_transform(torch.where(past_outcomes_squ == 0, past_emb, 0))
            
        return past_emb_trans, curr_match_emb

    def _att_layer(self, past_matches_emb, curr_match_emb, mask): 
        att = self.att(past_matches_emb, curr_match_emb, mask)
        
        return self.dropout(att)

    def _calc_mask(self, past_matches, curr_match): 
        """Calculates a mask for past_matches with time=0 (padding) or time equal to 
        current match time.
        Args:
            past_matches: Tensor of shape (#batches, history_len, 3). 
            curr_match: Tensor of shape (#batches, 2).
        Returns:
            mask: bool Tensor of shape (#batches, history_len)."""

        history_len = past_matches.shape[1]
        curr_match_time = torch.unsqueeze(curr_match[:, 1], -1).expand(-1, history_len) # Shape (batches, history_len, 1)
        mask = (curr_match_time == past_matches[:, :, 2]) + (past_matches[:, :, 2] == 0) # time = 0 (pad indicator), time = curr_match_time (mask curr match from history)

        return mask
        

def test():
    return

if __name__ == '__main__':
    test()