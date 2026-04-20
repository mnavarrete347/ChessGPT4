private static final int MEMORY_SIZE = 8;
private static final int[] moveMemory = new int[MEMORY_SIZE];
private static int memoryIndex = 0;

void main() throws Exception {
    System.out.println("info string user.dir=" + System.getProperty("user.dir"));
    System.out.println("info string MOVE_MAP_PATH=" + Constants.MOVE_MAP_PATH);
    System.out.println("info string MODEL_PATH=" + Constants.MODEL_PATH);
    System.out.println("info string moveMapExists=" + new java.io.File(Constants.MOVE_MAP_PATH).exists());
    System.out.println("info string modelExists=" + new java.io.File(Constants.MODEL_PATH).exists());

    //debugKiwipeteRoot();
    //debugKiwipeteDivideDepth2();

    BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);

    Position pos = Position.startPos();
    NeuralEngine nn = tryLoadNeuralEngine();
    if (nn != null) {
        System.out.println("info string Neural Engine loaded.");
    } else {
        System.out.println("info string Neural Engine unavailable.");
    }
    Position posAfterOurMove = null;

    // Active guess table populated while the opponent thinks.
    // Null when the engine has not yet played its first move, or NN is absent.
    Search.GuessTable activeGuessTable = null;
    Search.GuessingThread activeGuessingThread = null;

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

            // 1. Resolve Guessing Table
            if (opponentMove != 0 && activeGuessTable != null) {
                nnHint = activeGuessTable.lookupReply(opponentMove);
                if (nnHint != 0) {
                    System.out.println("info string Guess hit! opp=" + Move.toUci(opponentMove) + " hint=" + Move.toUci(nnHint));
                }
                else {
                    System.out.println("info string Guess miss for opp=" + Move.toUci(opponentMove));
                }
            }

            // 2. Perform Search
            int finalMove = freshSearch(pos, nnHint);

            // 3. Forced Variation on Repetition
            if (isRepeatingPattern() && finalMove != 0) {
//                System.out.println("info string Repetition detected, forcing variation.");
                // Re-run a very short search without the hint to find an alternative
                Search.startTime = System.currentTimeMillis();
                Search.timeLimit = 500;
                finalMove = Search.iterativeNegamax(pos, 0);
            }

            // 4. Output and Update
            if (finalMove == 0) {
                out.println("bestmove 0000");
            } else {
                out.println("bestmove " + Move.toUci(finalMove));
                recordMove(finalMove);
                // Crucial: Only update internal state if a valid move exists
                posAfterOurMove = pos.makeMove(finalMove);
                pos = posAfterOurMove;
            }

            // 5. Start Background Thinking
            if (nn != null && finalMove != 0) {
                System.out.println("info string Starting guess thread after " + Move.toUci(finalMove));
                activeGuessTable = new Search.GuessTable();
                activeGuessingThread = new Search.GuessingThread(posAfterOurMove, nn, activeGuessTable);
                activeGuessingThread.start();
            }

        } else if (line.startsWith("perft")) {
            String[] parts = line.split("\\s+");
            int depth = 1;
            if (parts.length >= 2) depth = Integer.parseInt(parts[1]);
            runPerft(pos, depth);
        } else if (line.equals("quit")) {
            stopGuessing(activeGuessingThread, activeGuessTable);
            break;
        }
    }
    out.flush();
}

// Signals the guessing thread to stop and waits briefly for it to exit.
private static void stopGuessing(Search.GuessingThread thread, Search.GuessTable table) {
    if (table != null) table.finish();
    if (thread != null) {
        try {
            thread.join(300);
        } catch (InterruptedException ignored) {
        }
    }
}

// Sets up timing and runs the parallel search.
// nnHint (may be 0) is forwarded to both the negamax and NN threads.
private static int freshSearch(Position pos, int nnHint) {
    Search.startTime = System.currentTimeMillis();
    Search.timeLimit = (long) (Search.moveTimeMs * 0.85);
    return Search.iterativeNegamax(pos, nnHint);
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

    if (i < tokens.length && tokens[i].equals("moves"))
        for (i++; i < tokens.length; i++) pos = pos.makeMove(Move.fromUci(tokens[i]));
    return pos;
}

static void parseGo(String cmd) {
    String[] tokens = cmd.trim().split("\\s+");
    Search.moveTimeMs = 10_000;
    Search.maxDepth = 100;
    for (int i = 1; i < tokens.length; i++) {
        switch (tokens[i].toLowerCase()) {
            case "movetime" -> {
                if (i + 1 < tokens.length) Search.moveTimeMs = Long.parseLong(tokens[++i]);
            }
            case "depth" -> {
                if (i + 1 < tokens.length) Search.maxDepth = Integer.parseInt(tokens[++i]);
            }
        }
    }
}

// -------------------------------------------------------------------------
// Move memory (repetition detection)
// -------------------------------------------------------------------------

static void recordMove(int move) {
    moveMemory[memoryIndex++ % MEMORY_SIZE] = move;
}

static boolean isRepeatingPattern() {
    if (memoryIndex < 6) return false;
    int a = moveMemory[(memoryIndex - 1) % MEMORY_SIZE];
    int b = moveMemory[(memoryIndex - 2) % MEMORY_SIZE];
    int c = moveMemory[(memoryIndex - 3) % MEMORY_SIZE];
    int d = moveMemory[(memoryIndex - 4) % MEMORY_SIZE];
    return a == c && b == d;
}

// -------------------------------------------------------------------------
// Perft
// -------------------------------------------------------------------------

static void runPerft(Position pos, int depth) {
    long start = System.nanoTime(), nodes = perft(pos, depth), end = System.nanoTime();
    double secs = (end - start) / 1e9;
    System.out.printf("Depth %d: %d nodes in %.3f s (NPS: %d)%n", depth, nodes, secs, (long) (nodes / secs));
}

static long perft(Position pos, int depth) {
    if (depth == 0) return 1;
    long nodes = 0;
    MoveList moves = pos.legalMoves();
    for (int i = 0; i < moves.count; i++) nodes += perft(pos.makeMove(moves.moves[i]), depth - 1);
    return nodes;
}

// -------------------------------------------------------------------------
// Neural engine loader
// -------------------------------------------------------------------------

@SuppressWarnings("unchecked")
private static NeuralEngine tryLoadNeuralEngine() {
    try {
        Map<String, Integer> moveMap;
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(Constants.MOVE_MAP_PATH))) {
            moveMap = (Map<String, Integer>) ois.readObject();
        }
        NeuralEngine engine = new NeuralEngine(Constants.MODEL_PATH, moveMap);
        //System.out.println("info string Neural Engine loaded.");
        return engine;
    } catch (Exception e) {
        System.out.println("info string Neural Engine unavailable: " + e);
        return null;
    }
}


static void debugKiwipeteRoot() {
    Position pos = Position.fromFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    MoveList moves = pos.legalMoves();
    System.out.println("Kiwipete legal move count = " + moves.count);
    for (int i = 0; i < moves.count; i++) {
        System.out.println(Move.toUci(moves.moves[i]));
    }
}

static void debugKiwipeteDivideDepth2() {
    Position pos = Position.fromFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    MoveList moves = pos.legalMoves();

    long total = 0;
    System.out.println("Kiwipete divide depth 2:");
    for (int i = 0; i < moves.count; i++) {
        int move = moves.moves[i];
        Position child = pos.makeMove(move);
        long count = perft(child, 1);   // depth 2 total = sum of each child’s legal moves
        total += count;
        System.out.println(Move.toUci(move) + " " + count);
    }
    System.out.println("Total " + total);
}