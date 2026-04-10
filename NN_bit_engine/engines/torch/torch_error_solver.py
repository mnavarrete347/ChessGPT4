import sys
import os
# Check if a local 'torch' is interfering
print(f"Current Directory: {os.getcwd()}")
try:
    import torch
    print(f"Torch Location: {torch.__file__}")
except AttributeError as e:
    print(f"Error caught: {e}")

print(torch.__version__)
print(torch.version.cuda)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))