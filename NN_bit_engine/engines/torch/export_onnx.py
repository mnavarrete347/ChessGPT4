import argparse
import pickle
import torch
import torch.nn as nn
from model import ChessModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export ChessModel (.pth) -> ONNX")
    parser.add_argument("--pth", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--mapping", required=True, help="Path to move_to_int pickle")
    parser.add_argument("--out", required=True, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def load_model(pth_path, num_classes):
    device = torch.device("cpu")

    model = ChessModel(num_classes=num_classes)
    checkpoint = torch.load(pth_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported checkpoint format")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # ── Load move mapping ─────────────────────────────
    print(f"[INFO] Loading mapping: {args.mapping}")
    with open(args.mapping, "rb") as f:
        move_to_int = pickle.load(f)

    num_classes = len(move_to_int)
    print(f"[INFO] num_classes = {num_classes}")

    # ── Load model ───────────────────────────────────
    print(f"[INFO] Loading model: {args.pth}")
    model = load_model(args.pth, num_classes)
    print("[INFO] Model loaded successfully")

    # ── Create dummy input ───────────────────────────
    dummy_input = torch.zeros(1, 13, 8, 8, dtype=torch.float32)

    # ── Export ONNX ──────────────────────────────────
    print(f"[INFO] Exporting ONNX to: {args.out}")

    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,

        input_names=["board_input"],
        output_names=["move_logits"],

        dynamic_axes={
            "board_input": {0: "batch_size"},
            "move_logits": {0: "batch_size"},
        }
    )

    print("[INFO] ONNX export complete")

    # ── Validate ONNX ────────────────────────────────
    try:
        import onnx
        model_onnx = onnx.load(args.out)
        onnx.checker.check_model(model_onnx)
        print("[INFO] ONNX model validation PASSED")
    except ImportError:
        print("[WARNING] onnx not installed, skipping validation")
    except Exception as e:
        print(f"[WARNING] ONNX validation failed: {e}")


if __name__ == "__main__":
    main()
