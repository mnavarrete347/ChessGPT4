private static final int MEMORY_SIZE = 8;
private static final int[] moveMemory = new int[MEMORY_SIZE];
private static int memoryIndex = 0;

void main() throws Exception {
    BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);

    Position pos = Position.startPos();
    NeuralEngine nn = tryLoadNeuralEngine();
    Position posAfterOurMove = null;

    Search.GuessTable activeGuessTable = null;
    Search.GuessingThread activeGuessingThread = null;

    if (nn == null) {
        System.out.println("info string nn unavailable");
    }

    String line;
    while ((line = in.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty()) continue;

        if (line.equals("uci")) {
            out.println("id name team_4_engine");
            out.println("id author team_4_kaung_martin_daniel_kevyn_victor");
            out.println("uciok");

        } else if (line.equals("isready")) {
            out.println("readyok");

        } else if (line.equals("ucinewgame")) {
            stopGuessing(activeGuessingThread, activeGuessTable);
            activeGuessingThread = null;
            activeGuessTable = null;
            pos = Position.startPos();
            posAfterOurMove = null;
            memoryIndex = 0;

        } else if (line.startsWith("position")) {
            pos = parsePosition(line, pos);

        } else if (line.startsWith("go")) {
            parseGo(line);
            stopGuessing(activeGuessingThread, activeGuessTable);

            int opponentMove = Search.detectOpponentMove(posAfterOurMove, pos);
            int nnHint = 0;

            if (opponentMove != 0 && activeGuessTable != null) {
                nnHint = activeGuessTable.lookupReply(opponentMove);
                if (nnHint != 0) {
//                    System.out.println("info string guess hit opp="
//                            + Move.toUci(opponentMove) + " hint=" + Move.toUci(nnHint));
                }
            }

            int finalMove = freshSearch(pos, nnHint, nn);

            if (isRepeatingPattern() && finalMove != 0) {
                int alternative = Search.findBestNonRepeatingMove(pos, finalMove, nnHint, nn);
                if (alternative != 0 && alternative != finalMove) {
//                    System.out.println("info string repetition avoid old="
//                            + Move.toUci(finalMove) + " new=" + Move.toUci(alternative));
                    finalMove = alternative;
                }
            }

            if (finalMove == 0) {
                out.println("bestmove 0000");
            } else {
                out.println("bestmove " + Move.toUci(finalMove));
                recordMove(finalMove);
                posAfterOurMove = pos.makeMove(finalMove);
                pos = posAfterOurMove;
            }

            if (nn != null && finalMove != 0) {
                activeGuessTable = new Search.GuessTable();
                activeGuessingThread = new Search.GuessingThread(posAfterOurMove, nn, activeGuessTable);
                activeGuessingThread.start();
            }

        } else if (line.startsWith("perft")) {
            String[] parts = line.split("\\s+");
            int depth = 1;
            if (parts.length >= 2) depth = Integer.parseInt(parts[1]);
            runPerft(pos, depth, out);

        } else if (line.equals("quit")) {
            stopGuessing(activeGuessingThread, activeGuessTable);
            break;
        }
    }
    out.flush();
}

private static void stopGuessing(Search.GuessingThread thread, Search.GuessTable table) {
    if (table != null) table.finish();
    if (thread != null) {
        try {
            thread.join(300);
        } catch (InterruptedException ignored) {
        }
    }
}

private static int freshSearch(Position pos, int nnHint, NeuralEngine nn) {
    Search.startTime = System.currentTimeMillis();
    Search.timeLimit = (long) (Search.moveTimeMs * 0.98);
    return Search.iterativeNegamax(pos, nnHint, nn);
}

// -------------------------------------------------------------------------
// UCI parsing
// -------------------------------------------------------------------------

static Position parsePosition(String cmd, Position currentPos) {
    String[] tokens = cmd.split("\\s+");
    int i = 1;
    Position pos = currentPos;

    if (i < tokens.length && tokens[i].equals("startpos")) {
        pos = Position.startPos();
        i++;
    } else if (i < tokens.length && tokens[i].equals("fen")) {
        i++;
        StringBuilder fen = new StringBuilder();
        for (int k = 0; k < 6 && i < tokens.length; k++, i++) {
            if (k > 0) fen.append(' ');
            fen.append(tokens[i]);
        }
        pos = Position.fromFEN(fen.toString());
    }

    if (i < tokens.length && tokens[i].equals("moves")) {
        for (i++; i < tokens.length; i++) pos = pos.makeMove(Move.fromUci(tokens[i]));
    }
    return pos;
}

static void parseGo(String cmd) {
    String[] tokens = cmd.trim().split("\\s+");
    Search.moveTimeMs = 10_000;
    Search.MAX_DEPTH = 100;

    for (int i = 1; i < tokens.length; i++) {
        switch (tokens[i].toLowerCase()) {
            case "movetime" -> {
                if (i + 1 < tokens.length) Search.moveTimeMs = Long.parseLong(tokens[++i]);
            }
            case "depth" -> {
                if (i + 1 < tokens.length) {
                    int requested = Integer.parseInt(tokens[++i]);
                    Search.MAX_DEPTH = Math.max(1, Math.min(64, requested));
                }
            }
        }
    }
}

static void recordMove(int move) {
    moveMemory[memoryIndex++ % MEMORY_SIZE] = move;
}

// Cheap ABAB repetition detector used only as a root-level escape hatch.
static boolean isRepeatingPattern() {
    if (memoryIndex < 6) return false;
    int a = moveMemory[(memoryIndex - 1) % MEMORY_SIZE];
    int b = moveMemory[(memoryIndex - 2) % MEMORY_SIZE];
    int c = moveMemory[(memoryIndex - 3) % MEMORY_SIZE];
    int d = moveMemory[(memoryIndex - 4) % MEMORY_SIZE];
    return a == c && b == d;
}

static void runPerft(Position pos, int depth, PrintWriter out) {
    long start = System.nanoTime(), nodes = perft(pos, depth), end = System.nanoTime();
    double secs = (end - start) / 1e9;
    out.printf("Depth %d: %d nodes in %.3f s (NPS: %d)%n", depth, nodes, secs, (long) (nodes / secs));
}

static long perft(Position pos, int depth) {
    if (depth == 0) return 1;
    long nodes = 0;
    MoveList moves = pos.legalMoves();
    for (int i = 0; i < moves.count; i++) nodes += perft(pos.makeMove(moves.moves[i]), depth - 1);
    return nodes;
}

@SuppressWarnings("unchecked")
private static NeuralEngine tryLoadNeuralEngine() {
    try {
        Map<String, Integer> moveMap;
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(Constants.MOVE_MAP_PATH))) {
            moveMap = (Map<String, Integer>) ois.readObject();
        }
        return new NeuralEngine(Constants.MODEL_PATH, moveMap);
    } catch (Exception e) {
        System.out.println("info string nn unavailable: " + e.getClass().getSimpleName());
        return null;
    }
}
