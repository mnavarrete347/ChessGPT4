import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chess import pgn, Board

from auxiliary_func import create_input_for_nn, encode_moves
from dataset import ChessDataset
from model import ChessModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PGN_DIR       = "../../data/pgn"
MODEL_OUT_DIR = "../../models"
LIMIT_GAMES   = 200_000
SAMPLE_LIMIT  = 5_000_000
BATCH_SIZE    = 1024
NUM_EPOCHS    = 150
LEARNING_RATE = 0.0001
LR_STEP_SIZE  = 20
LR_GAMMA      = 0.5
LOG_INTERVAL  = 1000
NUM_WORKERS   = 3

# Weight balancing the two loss terms.
# POLICY_LOSS_WEIGHT + VALUE_LOSS_WEIGHT should sum to 1.0.
# Start with policy weighted higher — move prediction is the harder task
# and the one your engine relies on most directly.
# If the value head is underfitting, increase VALUE_LOSS_WEIGHT slightly.
POLICY_LOSS_WEIGHT = 0.8
VALUE_LOSS_WEIGHT  = 0.2

# NOTE: AMP (mixed precision) is intentionally disabled.
# The Tesla P100 does not have fp16 tensor cores (that's Volta/Turing+),
# so autocast would be slower and less numerically stable on this hardware.
USE_AMP = False

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
print("CHESS ENGINE TRAINING SCRIPT (with Evaluation Head)")
print("=" * 60)

files = [os.path.join(PGN_DIR, f) for f in os.listdir(PGN_DIR) if f.endswith(".pgn")]
print(f"\nFound {len(files)} PGN file(s) in '{PGN_DIR}'")

print("\n--- Loading games ---", flush=True)
t0    = time.time()
games = load_games_with_limit(files, LIMIT_GAMES)
print(f"Loaded {len(games)} games in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. Convert to arrays
# ---------------------------------------------------------------------------
print(f"\n--- Converting games to board matrices (cap: {SAMPLE_LIMIT:,}) ---", flush=True)
t0          = time.time()
X, y, outcomes = create_input_for_nn(games, SAMPLE_LIMIT)
print(f"Samples built: {len(y):,}  ({time.time() - t0:.1f}s)")

# Log outcome distribution so you can verify your PGN data is balanced
n_white = int((outcomes ==  1.0).sum())
n_black = int((outcomes == -1.0).sum())
n_draw  = int((outcomes ==  0.0).sum())
print(f"Outcome distribution — White: {n_white:,}  Black: {n_black:,}  Draw: {n_draw:,}", flush=True)

games = []  # release python-chess Game objects

print("\n--- Encoding moves ---", flush=True)
y, move_to_int = encode_moves(y)
num_classes    = len(move_to_int)
print(f"Unique move classes: {num_classes}")

# Zero-copy conversions — torch.from_numpy shares the numpy buffer
print("\n--- Converting to PyTorch tensors (zero-copy) ---", flush=True)
X        = torch.from_numpy(X)
y        = torch.from_numpy(y)
outcomes = torch.from_numpy(outcomes)   # float32, shape (N,)

# ---------------------------------------------------------------------------
# 3. Dataset / DataLoader / Model
# ---------------------------------------------------------------------------
print("\n--- Setting up dataset, dataloader, and model ---", flush=True)

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"
print(f"Using device: {device}")
if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

dataset    = ChessDataset(X, y, outcomes)
dataloader = DataLoader(
    dataset,
    batch_size         = BATCH_SIZE,
    shuffle            = True,
    num_workers        = NUM_WORKERS,
    pin_memory         = use_cuda,
    persistent_workers = NUM_WORKERS > 0,
)

model     = ChessModel(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# Policy loss: CrossEntropy over move classes
policy_criterion = nn.CrossEntropyLoss()

# Value loss: MSELoss between predicted eval score and actual outcome.
# The model outputs a raw scalar; MSE penalises large deviations from
# the true +1/0/-1 outcome score.
value_criterion  = nn.MSELoss()

total_batches = len(dataloader)
print(
    f"Batches per epoch: {total_batches:,}  |  "
    f"Batch size: {BATCH_SIZE}  |  "
    f"Epochs: {NUM_EPOCHS}  |  "
    f"Workers: {NUM_WORKERS}  |  "
    f"AMP: {USE_AMP}  |  "
    f"Policy/Value loss weights: {POLICY_LOSS_WEIGHT}/{VALUE_LOSS_WEIGHT}",
    flush=True
)

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
print("\n--- Starting training ---", flush=True)
print("=" * 60)

training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start          = time.time()
    model.train()
    running_loss         = 0.0
    running_policy_loss  = 0.0
    running_value_loss   = 0.0

    for batch_idx, (inputs, labels, batch_outcomes) in enumerate(dataloader, start=1):
        inputs         = inputs.to(device, non_blocking=True)
        labels         = labels.to(device, non_blocking=True)
        batch_outcomes = batch_outcomes.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            policy_out, value_out = model(inputs)

            policy_loss = policy_criterion(policy_out, labels)
            value_loss  = value_criterion(value_out, batch_outcomes)

            # Combined weighted loss — both heads train together via one
            # backward pass through the shared backbone
            loss = POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss        += loss.item()
        running_policy_loss += policy_loss.item()
        running_value_loss  += value_loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            elapsed = time.time() - epoch_start
            pct     = 100.0 * batch_idx / total_batches
            print(
                f"  Epoch {epoch}/{NUM_EPOCHS} | "
                f"Batch {batch_idx:,}/{total_batches:,} ({pct:.1f}%) | "
                f"Loss: {running_loss / batch_idx:.4f}  "
                f"[policy: {running_policy_loss / batch_idx:.4f}  "
                f"value: {running_value_loss / batch_idx:.4f}] | "
                f"Elapsed: {elapsed:.0f}s",
                flush=True
            )

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    epoch_time = time.time() - epoch_start
    minutes    = int(epoch_time // 60)
    seconds    = int(epoch_time) % 60

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} complete | "
        f"Loss: {running_loss / total_batches:.4f}  "
        f"[policy: {running_policy_loss / total_batches:.4f}  "
        f"value: {running_value_loss / total_batches:.4f}] | "
        f"LR: {current_lr:.8f} | "
        f"Time: {minutes}m{seconds}s",
        flush=True
    )

total_time = time.time() - training_start
print("=" * 60)
print(f"Training finished in {total_time / 60:.1f} minutes.")

# ---------------------------------------------------------------------------
# 5. Save .pth, move mapping, and .onnx
# ---------------------------------------------------------------------------
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

base_name = (
    f"TORCH_EVH_{NUM_EPOCHS}EP_"
    f"{SAMPLE_LIMIT // 1_000_000}M{(SAMPLE_LIMIT % 1_000_000) // 100_000}SMPL_"
    f"{LIMIT_GAMES // 1000}KGM"
)
model_path   = os.path.join(MODEL_OUT_DIR, base_name + ".pth")
onnx_path    = os.path.join(MODEL_OUT_DIR, base_name + ".onnx")
mapping_path = os.path.join(MODEL_OUT_DIR, f"move_to_int_{base_name.replace('TORCH_', '')}")

print(f"\n--- Saving PyTorch model to: {model_path} ---", flush=True)
torch.save(model.state_dict(), model_path)
print("PyTorch model saved.")

print(f"--- Saving move mapping to: {mapping_path} ---", flush=True)
with open(mapping_path, "wb") as f:
    pickle.dump(move_to_int, f)
print("Move mapping saved.")

# Export to ONNX for use in your Java engine.
# The model has two outputs: policy logits and value scalar.
# In Java (ONNX Runtime) read them as:
#   OnnxTensor[] outputs = session.run(inputs);
#   float[] policyLogits = (float[]) outputs[0].getValue();  // shape [1, num_classes]
#   float   valueScore   = ((float[][]) outputs[1].getValue())[0][0]; // shape [1]
# Apply tanh to valueScore in Java: Math.tanh(valueScore)
# to map it to the [-1, 1] range for display/use.
print(f"\n--- Exporting ONNX model to: {onnx_path} ---", flush=True)
model.eval()
dummy_input = torch.zeros(1, 13, 8, 8, dtype=torch.float32).to(device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params    = True,
    opset_version    = 17,
    do_constant_folding = True,
    input_names      = ["board_state"],
    output_names     = ["policy_logits", "value"],
    dynamic_axes     = {
        "board_state":   {0: "batch_size"},
        "policy_logits": {0: "batch_size"},
        "value":         {0: "batch_size"},
    },
)
print("ONNX model saved.")

print("\nAll done.")
