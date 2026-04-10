import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from chess import pgn
from tqdm import tqdm

# Custom project imports
from auxiliary_func import create_input_for_nn, encode_moves
from dataset import ChessDataset
from model import ChessModel

def load_games_with_limit(file_paths, max_games):
    """
    Reads PGN files and returns a list of game objects up to a specified limit.
    """
    all_games = []
    
    for file_path in tqdm(file_paths, desc="Processing Files"):
        with open(file_path, 'r') as pgn_file:
            while len(all_games) < max_games:
                game = pgn.read_game(pgn_file)
                
                if game is None:  # End of current file
                    break
                
                all_games.append(game)
                
        if len(all_games) >= max_games:
            print(f"\nReached limit of {max_games} games. Stopping.")
            break
            
    return all_games

def train():
    # --- Configuration ---
    pgn_dir = "../../data/pgn"
    LIMIT_GAMES = 50000
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 100
    model_save_path = "../../models/TORCH_100EP_50KGM.pth"
    mapping_save_path = "../../models/move_to_int_100EP_50KGM"
    print("Configurations done!")
	
    # --- Data Loading ---
    files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith(".pgn")]
    games = load_games_with_limit(files, LIMIT_GAMES)

    # --- Preprocessing ---
    X, y = create_input_for_nn(games)
    print(f"NUMBER OF SAMPLES: {len(y)}")

    # Clear memory and slice (matches notebook logic)
    games = [] 
    X = X[0:2500000]
    y = y[0:2500000]

    y, move_to_int = encode_moves(y)
    num_classes = len(move_to_int)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # --- Model Setup ---
    dataset = ChessDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = ChessModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    # Note: Notebook uses 51-101 range for epoch labels
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
        end_time = time.time()
        epoch_time = end_time - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs + 1}, Loss: {avg_loss:.4f}, Time: {minutes}m{seconds}s')

    # --- Saving ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    with open(mapping_save_path, "wb") as file:
        pickle.dump(move_to_int, file)
    print(f"Move mapping saved to {mapping_save_path}")

if __name__ == "__main__":
    train()
