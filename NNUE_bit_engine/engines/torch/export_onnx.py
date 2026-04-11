"""
export_onnx.py
--------------
Converts a saved .pth checkpoint into a .onnx file for use in the Java engine.

Usage:
    python export_onnx.py \
        --pth  ../../models/TORCH_100EP_5M0SMPL_200KGM.pth \
        --mapping ../../models/move_to_int_100EP_5M0SMPL_200KGM \
        --out  ../../models/TORCH_100EP_5M0SMPL_200KGM.onnx

The script infers num_classes from the saved move mapping so you never have
to hard-code it or keep it in sync manually.

Java ONNX Runtime usage reminder
---------------------------------
The exported model has:
  Input  : "board_state"   shape [batch, 13, 8, 8]  float32
  Output : "policy_logits" shape [batch, num_classes] float32  (raw logits)
  Output : "value"         shape [batch]              float32  (pre-tanh score)

In Java:
    float valueScore = outputTensor.getFloatBuffer().get(0);
    double eval = Math.tanh(valueScore);   // maps to [-1, 1]
"""

import argparse
import pickle
import torch
from model import ChessModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export ChessModel .pth -> .onnx")
    parser.add_argument("--pth",     required=True,  help="Path to the .pth checkpoint")
    parser.add_argument("--mapping", required=True,  help="Path to the move_to_int pickle file")
    parser.add_argument("--out",     required=True,  help="Destination path for the .onnx file")
    parser.add_argument("--opset",   type=int, default=17, help="ONNX opset version (default: 17)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load move mapping to determine num_classes ───────────────────────────
    print(f"Loading move mapping from: {args.mapping}", flush=True)
    with open(args.mapping, "rb") as f:
        move_to_int = pickle.load(f)
    num_classes = len(move_to_int)
    print(f"num_classes = {num_classes}", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint from: {args.pth}", flush=True)
    device = torch.device("cpu")   # export on CPU for maximum portability
    model  = ChessModel(num_classes=num_classes)
    state  = torch.load(args.pth, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Checkpoint loaded.", flush=True)

    # ── Export ────────────────────────────────────────────────────────────────
    dummy_input = torch.zeros(1, 13, 8, 8, dtype=torch.float32)

    print(f"Exporting ONNX model to: {args.out}", flush=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        export_params       = True,
        opset_version       = args.opset,
        do_constant_folding = True,
        input_names         = ["board_state"],
        output_names        = ["policy_logits", "value"],
        dynamic_axes        = {
            "board_state":   {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value":         {0: "batch_size"},
        },
    )
    print("ONNX export complete.", flush=True)

    # ── Quick sanity check ────────────────────────────────────────────────────
    try:
        import onnx
        model_onnx = onnx.load(args.out)
        onnx.checker.check_model(model_onnx)
        print("ONNX model check passed.", flush=True)
    except ImportError:
        print("onnx package not installed — skipping validation check.", flush=True)
    except Exception as e:
        print(f"ONNX validation warning: {e}", flush=True)


if __name__ == "__main__":
    main()
