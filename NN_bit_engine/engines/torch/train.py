import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess import pgn

from auxiliary_func import create_input_for_nn, encode_moves
from dataset import ChessDataset
from model import ChessModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PGN_DIR       = "../../data/pgn"
MODEL_OUT_DIR = "../../models"
LIMIT_GAMES   = 1_000 # 50_000
SAMPLE_LIMIT  = 500_000 # 2_500_000
BATCH_SIZE    = 64
NUM_EPOCHS    = 10
LEARNING_RATE = 0.0001
LOG_INTERVAL  = 500   # print a batch-level update every N batches

# ---------------------------------------------------------------------------
# 1. Load PGN games
# ---------------------------------------------------------------------------
def load_games_with_limit(file_paths, max_games):
    all_games = []
    total_files = len(file_paths)

    for file_idx, file_path in enumerate(file_paths, start=1):
        print(f"[{file_idx}/{total_files}] Reading: {file_path}", flush=True)
        with open(file_path, 'r') as pgn_file:
            while len(all_games) < max_games:
                game = pgn.read_game(pgn_file)
                if game is None:
                    break
                all_games.append(game)

        print(f"  -> Games loaded so far: {len(all_games)}", flush=True)

        if len(all_games) >= max_games:
            print(f"Reached limit of {max_games} games. Stopping.", flush=True)
            break

    return all_games


print("=" * 60)
print("CHESS ENGINE TRAINING SCRIPT")
print("=" * 60)

files = [os.path.join(PGN_DIR, f) for f in os.listdir(PGN_DIR) if f.endswith(".pgn")]
print(f"\nFound {len(files)} PGN file(s) in '{PGN_DIR}'")

print("\n--- Loading games ---")
t0 = time.time()
games = load_games_with_limit(files, LIMIT_GAMES)
print(f"Loaded {len(games)} games in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. Convert to tensors
# ---------------------------------------------------------------------------
print("\n--- Converting games to board matrices ---", flush=True)
t0 = time.time()
X, y = create_input_for_nn(games)
print(f"Total samples (all games): {len(y)}  ({time.time() - t0:.1f}s)")

games = []  # free memory

# Cap the dataset
X = X[:SAMPLE_LIMIT]
y = y[:SAMPLE_LIMIT]
print(f"Samples after capping at {SAMPLE_LIMIT:,}: {len(y)}")

print("\n--- Encoding moves ---", flush=True)
y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)
print(f"Unique move classes: {num_classes}")

print("\n--- Converting to PyTorch tensors ---", flush=True)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ---------------------------------------------------------------------------
# 3. Dataset / DataLoader / Model
# ---------------------------------------------------------------------------
print("\n--- Setting up dataset, dataloader, and model ---", flush=True)

dataset    = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model     = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

total_batches = len(dataloader)
print(f"Batches per epoch: {total_batches:,}  |  Batch size: {BATCH_SIZE}  |  Epochs: {NUM_EPOCHS}")

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
print("\n--- Starting training ---", flush=True)
print("=" * 60)

training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(dataloader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        # Mid-epoch heartbeat
        if batch_idx % LOG_INTERVAL == 0:
            avg_loss_so_far = running_loss / batch_idx
            elapsed         = time.time() - epoch_start
            pct             = 100.0 * batch_idx / total_batches
            print(
                f"  Epoch {epoch}/{NUM_EPOCHS} | "
                f"Batch {batch_idx:,}/{total_batches:,} ({pct:.1f}%) | "
                f"Avg loss: {avg_loss_so_far:.4f} | "
                f"Elapsed: {elapsed:.0f}s",
                flush=True
            )

    epoch_time  = time.time() - epoch_start
    avg_loss    = running_loss / total_batches
    minutes     = int(epoch_time // 60)
    seconds     = int(epoch_time) % 60

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} complete | "
        f"Loss: {avg_loss:.4f} | "
        f"Time: {minutes}m{seconds}s",
        flush=True
    )

total_time = time.time() - training_start
print("=" * 60)
print(f"Training finished in {total_time / 60:.1f} minutes.")

# ---------------------------------------------------------------------------
# 5. Save model and move mapping
# ---------------------------------------------------------------------------
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

model_filename   = f"TORCH_{NUM_EPOCHS}EP_{SAMPLE_LIMIT // 1_000_000}M{(SAMPLE_LIMIT % 1_000_000) // 100_000}SMPL_{LIMIT_GAMES // 1000}KGM.pth"
model_path       = os.path.join(MODEL_OUT_DIR, model_filename)
mapping_path     = os.path.join(MODEL_OUT_DIR, "heavy_move_to_int")

print(f"\n--- Saving model to: {model_path} ---", flush=True)
torch.save(model.state_dict(), model_path)
print("Model saved.")

print(f"--- Saving move mapping to: {mapping_path} ---", flush=True)
with open(mapping_path, "wb") as f:
    pickle.dump(move_to_int, f)
print("Move mapping saved.")

print("\nAll done.")
