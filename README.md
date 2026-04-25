# ChessGPT4
Chess AI for ECE4318 Software Engineering Project — Team 4

## Authors
Kaung · Martin · Daniel · Kevyn · Victor

---

## Overview

ChessGPT4 is a UCI-compatible chess engine that combines a classical search pipeline with neural-network move guidance. It communicates with standard chess GUIs such as Arena or Cute Chess over stdin/stdout using the Universal Chess Interface (UCI) protocol.

At a high level, the engine:

1. receives a position from the GUI
2. searches for the best move under the current limit
3. returns `bestmove`
4. uses the opponent's time to prepare likely replies for the next turn

---

## Main Flow

### 1. Initialization
On startup, the engine enters the UCI loop and waits for commands from the host GUI.

If the neural model files are available, the engine loads them so neural guidance can be used during move selection and background reply preparation.

### 2. Position Setup
The GUI sends either:
- `position startpos ...`
- or `position fen ...`

The engine rebuilds the current board state from that command and stores the resulting position as the starting point for the next search.

### 3. Search
When the GUI sends `go`, the engine:
- reads the requested search limit
- stops any previous background guessing work
- checks whether a precomputed reply exists for the opponent's most recent move
- runs the main search
- returns the selected move with `bestmove`

The main search uses iterative deepening and move ordering so that promising moves are explored earlier.

### 4. Background Guessing
After sending its move, the engine starts a low-priority background thread while the opponent is thinking.

That thread:
- considers likely opponent replies
- uses the neural network to suggest candidate answers
- refines those answers with a short search
- stores them for possible reuse on the next turn

If the opponent later plays one of those predicted moves, the stored reply can be reused as a hint for the next search.

### 5. Repetition Avoidance
Before finalizing a move, the engine checks whether the selected move would repeat a previously seen position. If needed, it switches to a reasonable alternative instead of launching an expensive second search.

---

## File Structure

| File | Responsibility |
|---|---|
| `Main.java` | UCI loop, game state updates, search startup, background guessing lifecycle |
| `Search.java` | Main search, move ordering, repetition handling, guess table, guessing thread |
| `NeuralEngine.java` | ONNX model loading and neural move guidance |
| `Position.java` | Board state, move generation, move application, attack checks |
| `Move.java` | Packed move representation and UCI conversion |
| `MoveList.java` | Fixed-capacity move buffer |
| `Constants.java` | Search constants, scoring tables, masks, and attack tables |

---

## Building and Running

**Requirements:** JDK 21+, ONNX Runtime jar, model files in `models/`.

```bash
# Compile
javac -cp "lib\onnxruntime-1.24.3.jar" -d bin src\*.java

# Run
java -cp "bin;lib\onnxruntime-1.24.3.jar" Main
```

Then connect the engine to a UCI-compatible GUI.

---

## UCI Commands Supported

| Command | Behaviour |
|---|---|
| `uci` | Returns engine ID and `uciok` |
| `isready` | Returns `readyok` |
| `ucinewgame` | Resets internal game state |
| `position [startpos\|fen ...] [moves ...]` | Sets up the current board |
| `go movetime N` | Searches for approximately `N` ms |
| `go depth N` | Searches to fixed depth |
| `perft N` | Runs move-generation testing |
| `quit` | Stops background work and exits |
