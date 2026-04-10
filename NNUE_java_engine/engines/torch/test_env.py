import torch
import chess

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

board = chess.Board()
print(board)
