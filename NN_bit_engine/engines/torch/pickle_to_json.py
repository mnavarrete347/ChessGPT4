import argparse
import pickle
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Convert pickle mapping to JSON")
    parser.add_argument("--input", required=True, help="Path to .pkl file")
    parser.add_argument("--output", required=True, help="Path to output .json file")
    parser.add_argument("--reverse", action="store_true", help="Also save reverse mapping")
    return parser.parse_args()


def convert_pickle_to_json(input_path, output_path, save_reverse=False):
    # ── Load pickle ─────────────────────────────────────
    print(f"[INFO] Loading pickle file: {input_path}")
    with open(input_path, "rb") as f:
        mapping = pickle.load(f)

    print(f"[INFO] Loaded {len(mapping)} entries")

    # ── Ensure JSON serializable ───────────────────────
    # Convert keys to strings (JSON requires string keys)
    mapping_json = {str(k): int(v) for k, v in mapping.items()}

    # ── Save main mapping ──────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(mapping_json, f, indent=4)

    print(f"[INFO] Saved JSON mapping to: {output_path}")

    # ── Optional: reverse mapping ──────────────────────
    if save_reverse:
        reverse_mapping = {str(v): str(k) for k, v in mapping.items()}

        reverse_path = output_path.replace(".json", "_reverse.json")

        with open(reverse_path, "w") as f:
            json.dump(reverse_mapping, f, indent=4)

        print(f"[INFO] Saved reverse mapping to: {reverse_path}")


def main():
    args = parse_args()

    convert_pickle_to_json(
        input_path=args.input,
        output_path=args.output,
        save_reverse=args.reverse
    )


if __name__ == "__main__":
    main()
