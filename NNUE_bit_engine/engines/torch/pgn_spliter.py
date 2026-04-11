import os


def split_pgn(input_file, max_size_mb=45):
    # Convert MB to bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    file_count = 1
    current_out_file = None
    current_out_path = ""

    # Get base name for output files (e.g., "games.pgn" -> "games")
    base_name = os.path.splitext(input_file)[0]

    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # A new game starts with the [Event tag
                # If we have an open file and it's over the limit, close it to start a new one
                if line.startswith('[Event '):
                    if current_out_file:
                        if os.path.getsize(current_out_path) > max_size_bytes:
                            current_out_file.close()
                            current_out_file = None
                            file_count += 1

                # If no file is open, create the next chunk
                if current_out_file is None:
                    current_out_path = f"{base_name}_part{file_count}.pgn"
                    print(f"Creating: {current_out_path}")
                    current_out_file = open(current_out_path, 'w', encoding='utf-8')

                current_out_file.write(line)

    finally:
        if current_out_file:
            current_out_file.close()

    print(f"Done! Split into {file_count} files.")


if __name__ == "__main__":
    # Change 'large_games.pgn' to your actual filename
    split_pgn('lichess_elite_2025-10.pgn')
    split_pgn('lichess_elite_2025-11.pgn')