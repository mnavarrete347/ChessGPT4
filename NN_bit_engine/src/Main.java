void main() throws Exception {
    System.out.println("info string user.dir=" + System.getProperty("user.dir"));
    System.out.println("info string MOVE_MAP_PATH=" + Constants.MOVE_MAP_PATH);
    System.out.println("info string MODEL_PATH=" + Constants.MODEL_PATH);
    System.out.println("info string moveMapExists=" + new java.io.File(Constants.MOVE_MAP_PATH).exists());
    System.out.println("info string modelExists=" + new java.io.File(Constants.MODEL_PATH).exists());

    BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);

    Position pos = Position.startPos();
    Search.clearHistory();
    Search.recordPosition(pos);

    NeuralEngine nn = tryLoadNeuralEngine();
    if (nn != null) {
        System.out.println("info string Neural Engine loaded.");
    } else {
        System.out.println("info string Neural Engine unavailable.");
    }
    Position posAfterOurMove = null;

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

            Search.clearHistory();
            Search.recordPosition(pos);
            Search.clearHashMoves();

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
                    System.out.println("info string Guess hit! opp=" + Move.toUci(opponentMove)
                            + " hint=" + Move.toUci(nnHint));
                } else {
                    System.out.println("info string Guess miss for opp=" + Move.toUci(opponentMove));
                }
            }

            int finalMove = freshSearch(pos, nnHint);

            if (finalMove != 0 && Search.detectThreeRepeats(pos, finalMove)) {
                System.out.println("info string Repetition risk detected, trying cheap alternative.");
                int alternative = Search.bestNonRepeatingMove(pos, finalMove, nnHint);
                if (alternative != 0 && alternative != finalMove) {
                    System.out.println("info string Repetition avoided: "
                            + Move.toUci(finalMove) + " -> " + Move.toUci(alternative));
                    finalMove = alternative;
                }
            }

            if (finalMove == 0) {
                out.println("bestmove 0000");
            } else {
                out.println("bestmove " + Move.toUci(finalMove));
                posAfterOurMove = pos.makeMove(finalMove);
                pos = posAfterOurMove;
                Search.recordPosition(pos);
            }

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

private static void stopGuessing(Search.GuessingThread thread, Search.GuessTable table) {
    if (table != null) table.finish();
    if (thread != null) {
        try {
            thread.join(300);
        } catch (InterruptedException ignored) {
        }
    }
}

private static int freshSearch(Position pos, int nnHint) {
    Search.startTime = System.currentTimeMillis();
    Search.timeLimit = (long) (Search.moveTimeMs * 0.9);
    return Search.iterativeNegamax(pos, nnHint);
}

// -------------------------------------------------------------------------
// UCI parsing
// -------------------------------------------------------------------------

static Position parsePosition(String cmd, Position currentPos) {
    String[] tokens = cmd.split("\\s+");
    int i = 1;
    Position pos = currentPos;

    Search.clearHistory();

    if (i < tokens.length && tokens[i].equals("startpos")) {
        pos = Position.startPos();
        Search.recordPosition(pos);
        i++;
    } else if (i < tokens.length && tokens[i].equals("fen")) {
        i++;
        StringBuilder fen = new StringBuilder();
        for (int k = 0; k < 6 && i < tokens.length; k++, i++) {
            if (k > 0) fen.append(' ');
            fen.append(tokens[i]);
        }
        pos = Position.fromFEN(fen.toString());
        Search.recordPosition(pos);
    }

    if (i < tokens.length && tokens[i].equals("moves")) {
        for (i++; i < tokens.length; i++) {
            pos = pos.makeMove(Move.fromUci(tokens[i]));
            Search.recordPosition(pos);
        }
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
                    Search.MAX_DEPTH = Math.max(1, Math.min(64, Integer.parseInt(tokens[++i])));
                }
            }
        }
    }
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
    for (int i = 0; i < moves.count; i++) {
        nodes += perft(pos.makeMove(moves.moves[i]), depth - 1);
    }
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
        System.out.println("info string Neural Engine unavailable: " + e);
        return null;
    }
}