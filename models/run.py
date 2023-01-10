import os
import torch
import argparse
import trainer
import datasets
import models


def main(argp):
    if argp.model == 'player_history':
        train_history_model(argp.db_path, argp.save_path, argp.epochs)
    elif argp.model == 'combined':
        train_combined_model(argp.db_path, argp.save_path, argp.epochs)
    elif argp.model == 'team_comp':
        train_team_comp_model(argp.db_path, argp.save_path, argp.epochs)


def train_combined_model(db_path, save_path, num_epochs):
    # Hyperparameters
    regions = ['euw']
    patches = ['12.21']
    history_len = 40
    h_embed_dim = 40
    h_num_heads = 4
    tc_embed_dim = 12
    tc_num_heads = 2
    d_prob = 0.3
    batch_size = 128
    lr = 0.002
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")
        
    # Load datasets
    conn = datasets.SQLConnection(db_path, patches=patches, regions=regions) # Replace with argparse when finished testing
    match_dataset = datasets.MatchDataset(device, conn, champ_only=False)
    player_history = datasets.PlayerHistory(match_dataset, history_len=history_len)
    
    # Load model
    model = models.CombinedTransformerModel(player_history, match_dataset.num_champs, 
        h_embed_dim, h_num_heads, tc_embed_dim, tc_num_heads, d_prob).to(device)
        
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')

    # Load trainer
    train = trainer.Trainer(model, match_dataset, lr=lr)
    train.train(num_epochs=num_epochs, batch_size=batch_size)

    torch.save(model.state_dict(), os.path.join(save_path, 'combined_model_params.txt'))

    
def train_history_model(db_path, save_path, num_epochs):
    # Hyperparameters
    regions = ['euw'] 
    patches = ['12.21']
    history_len = 40
    embed_dim = 40
    num_heads = 4
    d_prob = 0.3
    batch_size = 128
    lr = 0.002
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")
        
    # Load datasets
    conn = datasets.SQLConnection(db_path, patches=patches, regions=regions) # Replace with argparse when finished testing
    match_dataset = datasets.MatchDataset(device, conn, champ_only=False)
    player_history = datasets.PlayerHistory(match_dataset, history_len=history_len)
    
    # Load model
    model = models.HistoryTransformerModel(player_history, num_champs=match_dataset.num_champs,
                                           embed_dim=embed_dim, num_heads=num_heads, d_prob=d_prob).to(device)  # Replace with argparse

    # Load trainer
    train = trainer.Trainer(model, match_dataset, lr=lr)
    train.train(num_epochs=num_epochs, batch_size=batch_size)

    torch.save(model.state_dict(), os.path.join(save_path, 'history_model_params.txt'))
    

def train_team_comp_model(db_path, save_path, num_epochs):
    # Hyperparameters
    regions = ['euw']
    patches = ['12.21']
    embed_dim = 20
    num_heads = 4
    d_prob = 0.4
    batch_size = 128
    lr = 0.001
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")
        
    # Load datasets
    conn = datasets.SQLConnection(db_path, patches=patches, regions=regions) # Replace with argparse when finished testing
    match_dataset = datasets.MatchDataset(device, conn, champ_only=True)
    
    # Load model
    model = models.TeamCompTransformerModel(num_champs=match_dataset.num_champs, embed_dim=embed_dim,
                                            num_heads=num_heads, d_prob=d_prob).to(device)

    # Load trainer
    train = trainer.Trainer(model, match_dataset, lr=lr)
    train.train(num_epochs=num_epochs, batch_size=batch_size)

    torch.save(model.state_dict(), os.path.join(save_path, 'team_comp_model_params.txt'))

    
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('model',
        help="Name of the model to train",
        choices=["team_comp", "player_history", "combined"])
    argp.add_argument('--db_path',
        help="path to the sqlite3 database containing the match data",
        default="data/matches.db")
    argp.add_argument('--save_path',
        help="path to save the model parameters",
        default="./")
    argp.add_argument('--epochs',
        help="path to save the model parameters",
        default="5")
    argp = argp.parse_args()
    argp.epochs = int(argp.epochs)
    
    main(argp)