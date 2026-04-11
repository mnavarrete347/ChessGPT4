import numpy as np
from chess import Board


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


# Maps the PGN [Result] header string to a float score.
# "*" means the game was unfinished/unknown — treated as a draw (0.0).
RESULT_MAP = {
    "1-0":  1.0,    # white wins
    "0-1": -1.0,    # black wins
    "1/2-1/2": 0.0, # draw
    "*":    0.0,    # unknown / unfinished
}


def parse_outcome(game) -> float:
    """
    Extract the game outcome from the PGN Result header.
    Returns +1.0 (white wins), -1.0 (black wins), or 0.0 (draw/unknown).
    """
    result = game.headers.get("Result", "*")
    return RESULT_MAP.get(result, 0.0)


def create_input_for_nn(games, sample_limit: int):
    """
    Convert games to (X, y, outcomes) arrays while capping at sample_limit.

    Returns:
        X        : float32 ndarray of shape (N, 13, 8, 8)
        y        : str     ndarray of shape (N,)  — UCI move strings
        outcomes : float32 ndarray of shape (N,)  — per-position game outcome
                   Every position in a game gets the same outcome value because
                   the result is known only at the end. The value head learns to
                   map board states to that eventual result.
    """
    X        = np.empty((sample_limit, 13, 8, 8), dtype=np.float32)
    y        = []
    outcomes = []
    idx      = 0

    for game in games:
        if idx >= sample_limit:
            break

        outcome = parse_outcome(game)
        board   = game.board()

        for move in game.mainline_moves():
            if idx >= sample_limit:
                break
            X[idx] = board_to_matrix(board)
            y.append(move.uci())
            outcomes.append(outcome)
            board.push(move)
            idx += 1

    X = X[:idx]
    return X, np.array(y), np.array(outcomes, dtype=np.float32)


def encode_moves(moves: np.ndarray):
    """Return integer-encoded moves and the move->int mapping."""
    unique_moves = list(dict.fromkeys(moves))
    move_to_int  = {m: i for i, m in enumerate(unique_moves)}
    encoded      = np.array([move_to_int[m] for m in moves], dtype=np.int64)
    return encoded, move_to_int
