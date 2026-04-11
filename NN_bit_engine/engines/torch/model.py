import torch.nn as nn


class ChessModel(nn.Module):
    """
    CNN with two output heads:

    1. Policy head  — predicts the next move (same as before).
                      Output shape: (batch, num_classes)  [raw logits]

    2. Value head   — predicts the game outcome from the current position.
                      Output shape: (batch, 1)  [single float in (-inf, +inf)]
                      Passed through tanh at inference so the result is in [-1, 1]:
                        +1  = white wins
                         0  = draw
                        -1  = black wins

    The two heads share the conv backbone and fc1, then branch at fc2.
    This means the conv filters learn features that are useful for BOTH
    move prediction and position evaluation simultaneously.
    """

    def __init__(self, num_classes: int):
        super(ChessModel, self).__init__()

        # ── Shared backbone ──────────────────────────────────────────────────
        self.conv1   = nn.Conv2d(13, 64,  kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(8 * 8 * 128, 256)
        self.relu    = nn.ReLU()

        # ── Policy head (move prediction) ────────────────────────────────────
        # Direct projection from the shared 256-dim representation.
        self.policy_fc = nn.Linear(256, num_classes)

        # ── Value head (position evaluation) ─────────────────────────────────
        # An extra hidden layer gives the value head its own capacity to learn
        # evaluation-specific features without interfering with the policy head.
        self.value_fc1 = nn.Linear(256, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # ── Weight initialisation ─────────────────────────────────────────────
        nn.init.kaiming_uniform_(self.conv1.weight,   nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight,   nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.policy_fc.weight)
        nn.init.kaiming_uniform_(self.value_fc1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.value_fc2.weight)

    def forward(self, x):
        # Shared backbone
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        # Policy head — raw logits (CrossEntropyLoss handles softmax internally)
        policy = self.policy_fc(x)

        # Value head — tanh squashes output to [-1, 1]
        value = self.relu(self.value_fc1(x))
        value = self.value_fc2(value)          # shape: (batch, 1)
        value = value.squeeze(1)               # shape: (batch,)

        return policy, value
