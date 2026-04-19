# ChessGPT4
Chess AI for ECE4318 Software Engineering Project — Team 4

## Authors
Kaung · Martin · Daniel · Kevyn · Victor

---

## Overview

ChessGPT4 is a UCI-compatible chess engine that combines a classical alpha-beta negamax search with an ONNX neural network. The engine communicates with any standard chess GUI (Arena, Cute Chess, etc.) over stdin/stdout using the Universal Chess Interface (UCI) protocol.

---

## Architecture

### Board Representation
The engine uses **bitboards** — one 64-bit integer per piece type per color (12 total). Legal move generation is performed by first generating pseudo-legal moves and filtering out those that leave the king in check. Move encoding is a packed integer: bits 0–5 = from square, 6–11 = to square, 12–14 = promotion piece.

### Evaluation
Positions are scored using **tapered piece-square tables** that interpolate between midgame and endgame values based on remaining material (the phase). In the endgame (total material below a threshold), a king centrality bonus is added to encourage active king play.

### Search
The search is **iterative-deepening negamax with alpha-beta pruning**. Move ordering uses:
- The previous iteration's best move (PV move) at priority 20000
- An optional NN hint move at priority 10000
- MVV-LVA for captures, flat bonus for promotions

The search runs for 85% of the allocated move time to leave a safety margin.

### Neural Network
An ONNX model provides two outputs:
- **Policy head** — logits over the full move vocabulary, used to select a hint move
- **Value head** — a scalar in (−1, +1) representing the expected outcome from the side to move's perspective

The network takes a `[1, 13, 8, 8]` input tensor: one binary plane per piece type (channels 0–11) plus a legal-move destination mask (channel 12).

ONNX Runtime's internal thread pools are capped to half the available logical cores to prevent inference calls from saturating the CPU.

### Background Guessing
After the engine plays its move, a low-priority daemon thread (`GuessingThread`) runs during the opponent's clock. It:
1. Ranks the opponent's legal moves by PST plausibility
2. For each top candidate, runs one NN forward pass to pre-compute our best reply
3. Stores up to `MAX_GUESSES` (opponent move → our reply) pairs in a `GuessTable`

When the next `go` command arrives, the engine detects the opponent's actual move by comparing board states, looks it up in the table, and if found, passes the pre-computed reply as an `nnHint` to the search — promoting it to the front of move ordering at every depth.

---

## File Structure

| File | Responsibility |
|---|---|
| `Main.java` | UCI loop, guessing thread lifecycle, repetition detection |
| `Search.java` | Negamax, move ordering, evaluation, `GuessTable`, `GuessingThread`, `detectOpponentMove` |
| `NeuralEngine.java` | ONNX session management, policy head, value head |
| `Position.java` | Immutable board state, `makeMove`, legal/pseudo-legal move generation, attack detection |
| `Move.java` | Packed-int move encoding and UCI string conversion |
| `MoveList.java` | Fixed-capacity move buffer (256 slots) |
| `Constants.java` | PSTs, material values, phase weights, bitboard masks, attack tables |
| `SearchResult.java` | Lightweight record: `(int move, int score)` |

---

## Building and Running

**Requirements:** JDK 21+, ONNX Runtime 1.24.3 (`lib/onnxruntime-1.24.3.jar`), engine model files in `models/`.

```bash
# Compile
javac -cp "lib\onnxruntime-1.24.3.jar" -d bin src\*.java

# Run (UCI mode — connect via Arena or Cute Chess)
java -cp "bin;lib\onnxruntime-1.24.3.jar" Main

# Test client
javac -d bin src\UciTestClient.java
java -cp bin UciTestClient
```

---

## UCI Commands Supported

| Command | Behaviour |
|---|---|
| `uci` | Returns engine ID and `uciok` |
| `isready` | Loads ONNX model, returns `readyok` |
| `ucinewgame` | Resets board, clears guessing state |
| `position [startpos\|fen ...] [moves ...]` | Sets up the board |
| `go movetime N` | Searches for `N` ms, outputs `bestmove` |
| `go depth N` | Searches to fixed depth |
| `perft` | Runs move-generation correctness test to depth 6 |
| `quit` | Stops guessing thread and exits |
