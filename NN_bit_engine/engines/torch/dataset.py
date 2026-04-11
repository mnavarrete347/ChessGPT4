from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """
    Dataset holding board states, move labels, and position outcomes.

    Args:
        X        : float32 tensor of shape (N, 13, 8, 8) — board matrices
        y        : long    tensor of shape (N,)           — encoded move indices
        outcomes : float32 tensor of shape (N,)           — game outcome per position
                     +1.0  white wins
                      0.0  draw
                     -1.0  black wins
    """

    def __init__(self, X, y, outcomes):
        self.X        = X
        self.y        = y
        self.outcomes = outcomes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.outcomes[idx]
