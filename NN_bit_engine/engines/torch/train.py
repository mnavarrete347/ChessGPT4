import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess import pgn, Board

from dataset import ChessDataset
from model import ChessModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PGN_DIR       = "../../data/pgn"
MODEL_OUT_DIR = "../../models"
LIMIT_GAMES   = 200_000   # was 50,000  — more game diversity = better model
SAMPLE_LIMIT  = 5_000_000 # was 2,500,000 — 32GB+ RAM can handle this
BATCH_SIZE    = 1024      # was 512 — P100 has 16GB VRAM, keep it fed
NUM_EPOCHS    = 100       # was 50 — previous model hadn't plateaued yet
LEARNING_RATE = 0.0001
LR_STEP_SIZE  = 20        # halve the LR every N epochs
LR_GAMMA      = 0.5       # LR multiplier on each step (0.0001 -> 0.00005 -> ...)
LOG_INTERVAL  = 500       # print a batch-level update every N batches
NUM_WORKERS   = 8         # was 4 — more CPU cores available on server

# NOTE: AMP (mixed precision) is intentionally disabled.
# The Tesla P100 does not have fp16 tensor cores (that's Volta/Turing+),
# so autocast would be slower and less numerically stable on this hardware.
USE_AMP = False

# ---------------------------------------------------------------------------
# Optimized helpers
# ---------------------------------------------------------------------------

def board_to_matrix(board: Board) -> np.ndarray:
    """Build a (13, 8, 8) float32 matrix directly — avoids a later cast."""
    matrix = np.zeros((13, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel  = (piece.piece_type - 1) + (0 if piece.color else 6)
        matrix[channel, row, col] = 1.0
    for move in board.legal_moves:
        row, col = divmod(move.to_square, 8)
        matrix[12, row, col] = 1.0
    return matrix


def create_input_for_nn(games, sample_limit: int):
    """
    Convert games to (X, y) arrays while capping at sample_limit.
    - Pre-allocates X to avoid the 2x RAM spike from list -> np.array conversion.
    - Stops early at sample_limit so we never build more data than needed.
    - Uses float32 throughout so no dtype cast is needed.
    """
    X   = np.empty((sample_limit, 13, 8, 8), dtype=np.float32)
    y   = []
    idx = 0

    for game in games:
        if idx >= sample_limit:
            break
        board = game.board()
        for move in game.mainline_moves():
            if idx >= sample_limit:
                break
            X[idx] = board_to_matrix(board)
            y.append(move.uci())
            board.push(move)
            idx += 1

    X = X[:idx]
    return X, np.array(y)


def encode_moves(moves: np.ndarray):
    """Return integer-encoded moves and the move->int mapping."""
    unique_moves = list(dict.fromkeys(moves))
    move_to_int  = {m: i for i, m in enumerate(unique_moves)}
    encoded      = np.array([move_to_int[m] for m in moves], dtype=np.int64)
    return encoded, move_to_int


# ---------------------------------------------------------------------------
# 1. Load PGN games
# ---------------------------------------------------------------------------

def load_games_with_limit(file_paths, max_games):
    all_games   = []
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

print("\n--- Loading games ---", flush=True)
t0    = time.time()
games = load_games_with_limit(files, LIMIT_GAMES)
print(f"Loaded {len(games)} games in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. Convert to arrays (RAM-efficient, capped during construction)
# ---------------------------------------------------------------------------
print(f"\n--- Converting games to board matrices (cap: {SAMPLE_LIMIT:,}) ---", flush=True)
t0   = time.time()
X, y = create_input_for_nn(games, SAMPLE_LIMIT)
print(f"Samples built: {len(y):,}  ({time.time() - t0:.1f}s)")

games = []   # release python-chess Game objects

print("\n--- Encoding moves ---", flush=True)
y, move_to_int = encode_moves(y)
num_classes    = len(move_to_int)
print(f"Unique move classes: {num_classes}")

# torch.from_numpy shares the underlying buffer — no second copy of X in RAM.
print("\n--- Converting to PyTorch tensors (zero-copy for X) ---", flush=True)
X = torch.from_numpy(X)
y = torch.from_numpy(y)

# ---------------------------------------------------------------------------
# 3. Dataset / DataLoader / Model
# ---------------------------------------------------------------------------
print("\n--- Setting up dataset, dataloader, and model ---", flush=True)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
print(f"Using device: {device}")
if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

dataset    = ChessDataset(X, y)
dataloader = DataLoader(
    dataset,
    batch_size         = BATCH_SIZE,
    shuffle            = True,
    num_workers        = NUM_WORKERS,
    pin_memory         = use_cuda,          # faster CPU->GPU transfers
    persistent_workers = NUM_WORKERS > 0,   # avoids worker restart cost per epoch
)

model     = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Halves the learning rate every LR_STEP_SIZE epochs:
#   Epoch  1-20:  lr = 0.0001
#   Epoch 21-40:  lr = 0.00005
#   Epoch 41-60:  lr = 0.000025
#   Epoch 61-80:  lr = 0.0000125
#   Epoch 81-100: lr = 0.00000625
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# GradScaler: disabled on P100 (no fp16 tensor cores), kept for portability
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

total_batches = len(dataloader)
print(
    f"Batches per epoch: {total_batches:,}  |  "
    f"Batch size: {BATCH_SIZE}  |  "
    f"Epochs: {NUM_EPOCHS}  |  "
    f"Workers: {NUM_WORKERS}  |  "
    f"AMP: {USE_AMP}",
    flush=True
)

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
print("\n--- Starting training ---", flush=True)
print("=" * 60)

training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start  = time.time()
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(dataloader, start=1):
        # non_blocking overlaps CPU->GPU transfer with compute (requires pin_memory)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # set_to_none=True skips the memset — faster than zeroing gradients
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

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

    # Step the scheduler at the end of each epoch
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    epoch_time = time.time() - epoch_start
    avg_loss   = running_loss / total_batches
    minutes    = int(epoch_time // 60)
    seconds    = int(epoch_time) % 60

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} complete | "
        f"Loss: {avg_loss:.4f} | "
        f"LR: {current_lr:.8f} | "
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

model_filename = (
    f"TORCH_{NUM_EPOCHS}EP_"
    f"{SAMPLE_LIMIT // 1_000_000}M{(SAMPLE_LIMIT % 1_000_000) // 100_000}SMPL_"
    f"{LIMIT_GAMES // 1000}KGM.pth"
)
map_filename = (
    f"move_to_int_{NUM_EPOCHS}EP_"
    f"{SAMPLE_LIMIT // 1_000_000}M{(SAMPLE_LIMIT % 1_000_000) // 100_000}SMPL_"
    f"{LIMIT_GAMES // 1000}KGM"
)
model_path   = os.path.join(MODEL_OUT_DIR, model_filename)
mapping_path = os.path.join(MODEL_OUT_DIR, map_filename)

print(f"\n--- Saving model to: {model_path} ---", flush=True)
torch.save(model.state_dict(), model_path)
print("Model saved.")

print(f"--- Saving move mapping to: {mapping_path} ---", flush=True)
with open(mapping_path, "wb") as f:
    pickle.dump(move_to_int, f)
print("Move mapping saved.")

print("\nAll done.")
