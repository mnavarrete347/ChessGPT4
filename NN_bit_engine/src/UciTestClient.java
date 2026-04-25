import static java.lang.Thread.sleep;

/**
* UciTestClient — comprehensive test harness for the chess engine.
* HOW TO COMPILE AND RUN
* ──────────────────────
*   javac -d bin src/UciTestClient.java
*   java  -cp bin UciTestClient
* Make sure the engine is already compiled first:
*   javac -cp "lib\onnxruntime-1.24.3.jar" -d bin src\*.java
* ADDING YOUR OWN TEST POSITIONS
* ───────────────────────────────
* Position tests live in buildTestSuite().
* Guess-table tests live in buildGuessTestSuite().
* Each type has its own section below with examples.
*/
// =========================================================================
// Engine launch configuration
// =========================================================================

    private static final String ENGINE_CLASS = "Main";
    //private static final String CLASS_PATH = "out/production/NN_bit_engine;ChessGPT4-main/NN_bit_engine/lib/onnxruntime-1.24.3.jar";
    private static final String CLASS_PATH = "bin;lib/onnxruntime-1.24.3.jar";

    // How long after movetime expires the client waits before giving up.
    private static final int WAIT_BUFFER_MS = 2000;
    // How long to wait between test cases.
    private static final int BETWEEN_TESTS_MS = 500;

    // =========================================================================
    // PositionTest — a single position with optional expected-move validation
    // =========================================================================

    static class PositionTest {
    final String description;
    final String fen;           // null when positionCommand() is overridden
    final int moveTimeMs;
    final String[] expectedMoves; // empty = any legal move is acceptable

    PositionTest(String description, String fen, int moveTimeMs, String... expectedMoves) {
        this.description = description;
        this.fen = fen;
        this.moveTimeMs = moveTimeMs;
        this.expectedMoves = expectedMoves;
    }

    // Factory for move-sequence tests: "position startpos moves a b c ..."
    static PositionTest fromMoves(String description, int moveTimeMs,
                                  String[] expectedMoves, String... moves) {
        return new PositionTest(description, null, moveTimeMs, expectedMoves) {
            @Override
            String positionCommand() {
                return "position startpos moves " + String.join(" ", moves);
            }
        };
    }

    String positionCommand() {
        return fen != null ? "position fen " + fen : "position startpos";
    }

    boolean validates(String bestMove) {
        if (expectedMoves.length == 0) return true;
        for (String e : expectedMoves) if (e.equalsIgnoreCase(bestMove)) return true;
        return false;
    }
    }

    // =========================================================================
    // GuessScenario — tests the engine's background guessing while waiting
    //
    // HOW IT WORKS IN THE ENGINE:
    //   After bestmove is sent, the engine calls GuessingThread which uses the
    //   NN policy head to generate up to 10 (opponentMove, ourReply) pairs while
    //   the opponent thinks. When the next "go" arrives, the engine calls
    //   detectOpponentMove() and looks the opponent's actual move up in the table.
    //   If found, parallelSearch receives that pre-computed reply as an nnHint.
    //
    // HOW WE TEST IT:
    //   We send two moves (first + second) to the engine as a two-turn sequence.
    //   Between them we wait long enough for the guessing thread to run.
    //   We then inspect the engine log for:
    //     - "guess[N]" lines  → guesses were actually generated
    //     - "Guess hit!"      → opponent's move was in the table (hint was used)
    //     - "Guess miss"      → opponent's move was not guessed
    //   We also measure whether the second search finishes faster than a cold
    //   search would, which indirectly validates that the hint improved ordering.
    //
    // ADDING YOUR OWN GUESS SCENARIOS
    // ────────────────────────────────
    //   new GuessScenario(
    //       "My scenario description",
    //       firstFen,     // position where engine plays move 1
    //       secondFen,    // position where opponent has replied (engine plays move 2)
    //       waitMs,       // time between move 1 and sending the second position
    //                     // (gives guessing thread time to run; 1000–3000 ms is good)
    //       moveTimeMs,   // budget for each search
    //       expectHit     // true if we expect the opponent's reply to be in the table
    //   )
    // =========================================================================

    /**
    * @param firstFen  engine thinks here, then guessing thread starts
    //* @param secondFen position after the opponent's actual reply
    * @param waitMs    how long to let the guessing thread run
    * @param expectHit whether we expect a "Guess hit!" in the log
    */
    record GuessScenario(String description, String firstFen, String opponentMoveUci,
             int waitMs, int moveTimeMs, boolean expectHit) {
    }

    // =========================================================================
    // TEST SUITE — position tests (add/remove freely)
    // =========================================================================

    static List<PositionTest> buildTestSuite() {
    List<PositionTest> suite = new ArrayList<>();

    suite.add(new PositionTest(
            "Starting position",
            buildFen("rnbqkbnr", "pppppppp", "8", "8", "8", "8", "PPPPPPPP", "RNBQKBNR"),
            10000
    ));

    suite.add(new PositionTest(
            "Scholar's mate threat — engine must defend f7",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            10000,
            "f7f6", "f7f5", "d8e7", "g8e7", "f6e4"
    ));

    suite.add(new PositionTest(
            "Endgame — king and pawn",
            "8/8/8/8/3k4/8/3P4/3K4 w - - 0 1",
            10000
    ));

    suite.add(new PositionTest(
            "King Safety - Castle to escape E-file pressure",
            "r3k3/ppp5/3q4/8/8/8/PPPP4/R3K3 w Qq - 0 1",
            10000,
            "e1c1"
    ));

    suite.add(new PositionTest(
            "Knight outpost — white knight on c3",
            buildFen("rnbqkbnr", "pppppppp", "8", "8", "8", "2N5", "PPPPPPPP", "R1BQKBNR"),
            10000
    ));

    suite.add(new PositionTest(
            "King in check — must respond legally",
            "8/8/8/8/8/8/6q1/6K1 w - - 0 1",
            10000,
            "g1h1", "g1f1", "g1g2"
    ));

    suite.add(PositionTest.fromMoves(
            "Ruy Lopez — engine plays as White after 1.e4 e5 2.Nf3 Nc6",
            10000,
            new String[]{"f1b5", "f1c4", "d2d4"},
            "e2e4", "e7e5", "g1f3", "b8c6"
    ));

    suite.add(PositionTest.fromMoves(
            "Sicilian — engine plays as Black after 1.e4 c5 2.Nf3",
            10000,
            new String[]{},
            "e2e4", "c7c5", "g1f3"
    ));

    return suite;
    }

    // =========================================================================
    // GUESS TEST SUITE — add/remove scenarios freely
    // =========================================================================

    static List<GuessScenario> buildGuessTestSuite() {
    List<GuessScenario> suite = new ArrayList<>();

    suite.add(new GuessScenario(
    "Startpos: opponent plays e7e5 (likely guess hit)",
    buildFen("rnbqkbnr", "pppppppp", "8", "8", "8", "8", "PPPPPPPP", "RNBQKBNR"),
    "e7e5",
    2000,
    4000,
    true
    ));

    suite.add(new GuessScenario(
    "Startpos: opponent plays h7h5 (unlikely — expect guess miss)",
    buildFen("rnbqkbnr", "pppppppp", "8", "8", "8", "8", "PPPPPPPP", "RNBQKBNR"),
    "h7h5",
    2000,
    4000,
    false
    ));

    suite.add(new GuessScenario(
    "Very short wait (100 ms) — guessing thread may still be filling",
    buildFen("rnbqkbnr", "pppppppp", "8", "8", "8", "8", "PPPPPPPP", "RNBQKBNR"),
    "e7e5",
    100,
    4000,
    false
    ));

    suite.add(new GuessScenario(
    "Full wait (3 s) — table should have multiple guesses",
    buildFen("r1bqkb1r", "pppp1ppp", "2n2n2", "4p3", "2B1P3", "5N2", "PPPP1PPP", "RNBQK2R"),
    "d7d5",
    3000,
    4000,
    true
    ));

    suite.add(new GuessScenario(
    "Endgame with few opponent moves — table fills fast",
    "8/8/8/8/3k4/8/3P4/3K4 w - - 0 1",
    "d4c4",
    1500,
    3000,
    false
    ));

    return suite;
    }

    // =========================================================================
    // Perft expected node counts
    // =========================================================================

    record PerftExpected(String description, String fen, int depth, long expectedNodes) {
    }

    static List<PerftExpected> buildPerftSuite() {
    List<PerftExpected> suite = new ArrayList<>();
    // Node counts assume ENABLE_EN_PASSANT = false.
        suite.add(new PerftExpected("Start pos depth 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20));
        suite.add(new PerftExpected("Start pos depth 2", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400));
        suite.add(new PerftExpected("Start pos depth 3", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902));
        suite.add(new PerftExpected("Start pos depth 4", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197_281));
        suite.add(new PerftExpected("Start pos depth 5", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4_865_351));
        suite.add(new PerftExpected("Start pos depth 6", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119_048_441));
        suite.add(new PerftExpected("Kiwipete depth 1", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48));
        suite.add(new PerftExpected("Kiwipete depth 2", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2038));
        return suite;
    }

    // =========================================================================
    // Test result tracking
    // =========================================================================

    record TestResult(String name, boolean passed, String detail) {
    }

    static final List<TestResult> results = Collections.synchronizedList(new ArrayList<>());

    // =========================================================================
    // Engine communication state — all written by the reader thread
    // =========================================================================

    static BufferedWriter writer;

    static final AtomicReference<String> lastBestMove = new AtomicReference<>("");
    static final AtomicLong bestMoveTimestamp = new AtomicLong(0);
    static final AtomicLong goTimestamp = new AtomicLong(0);
    static final AtomicLong lastPerftNodes = new AtomicLong(-1);
    static final List<String> engineLog = Collections.synchronizedList(new ArrayList<>());

    static volatile int awaitedPerftDepth = -1;

    // Guess-specific counters — reset before each guess scenario
    static final AtomicInteger guessCount = new AtomicInteger(0);
    static final AtomicBoolean sawGuessHit = new AtomicBoolean(false);
    static final AtomicBoolean sawGuessMiss = new AtomicBoolean(false);

    static volatile CountDownLatch bestMoveLatch = new CountDownLatch(1);
    static volatile CountDownLatch perftLatch = new CountDownLatch(1);

    // =========================================================================
    // Main
    // =========================================================================

    void main() throws Exception {
    header("UCI TEST CLIENT");
    IO.println("Engine: " + ENGINE_CLASS + "  |  CP: " + CLASS_PATH);

        try (Process process = new ProcessBuilder("java", "-cp", CLASS_PATH, ENGINE_CLASS)
                .redirectErrorStream(true)
                .start()) {

            writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            // Reader thread — monitors all engine output and updates shared state
            Thread readerThread = new Thread(() -> {
                try {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        engineLog.add(line);
                        IO.println("  ENGINE> " + line);

                        // bestmove
                        if (line.startsWith("bestmove")) {
                            String[] parts = line.split("\\s+");
                            lastBestMove.set(parts.length > 1 ? parts[1] : "");
                            bestMoveTimestamp.set(System.currentTimeMillis());
                            bestMoveLatch.countDown();
                        }

                        // Guess-table output from GuessingThread
                        // Engine logs:  info string guess[N] opp=<move> reply=<move>
                        if (line.contains("guess[")) {
                            guessCount.incrementAndGet();
                        }
                        // Engine logs:  info string Guess hit! opp=<move> hint=<move>
                        if (line.contains("Guess hit")) {
                            sawGuessHit.set(true);
                        }
                        // Engine logs:  info string Guess miss for opp=<move>
                        if (line.contains("Guess miss")) {
                            sawGuessMiss.set(true);
                        }

                        // Perft: "Depth N: X nodes in ..."
                        if (line.matches("Depth \\d+:.*nodes.*")) {
                            try {
                                int colon = line.indexOf(':');
                                int depth = Integer.parseInt(line.substring(6, colon).trim());

                                if (depth == awaitedPerftDepth) {
                                    String[] parts = line.replace(":", "").split("\\s+");
                                    for (int i = 0; i < parts.length; i++) {
                                        if (parts[i].equals("nodes") && i > 0) {
                                            lastPerftNodes.set(Long.parseLong(parts[i - 1]));
                                            perftLatch.countDown();
                                            break;
                                        }
                                    }
                                }
                            } catch (Exception ignored) {
                            }
                        }
                    }
                } catch (IOException e) {
                    IO.println("  [reader ended: " + e.getMessage() + "]");
                }
            }, "engine-reader");
            readerThread.setDaemon(true);
            readerThread.start();

            // Handshake
            header("HANDSHAKE");
            send("uci");
            send("isready");
            sleep(500);
            send("ucinewgame");
            sleep(300);

            // Run suites
            runPositionTests(buildTestSuite());
            //runGuessTests(buildGuessTestSuite());
            runPerftTests(buildPerftSuite());
            runStressTest();

            // Shutdown
            header("SHUTDOWN");
            send("quit");
            sleep(500);
            process.destroy();
        }

        printSummary();
    }

    // =========================================================================
    // Position test runner
    // =========================================================================

    static void runPositionTests(List<PositionTest> suite) throws Exception {
    header("POSITION TESTS  (" + suite.size() + " cases)");

    for (int i = 0; i < suite.size(); i++) {
        PositionTest test = suite.get(i);
        IO.println();
        subHeader((i + 1) + "/" + suite.size() + "  " + test.description);

        bestMoveLatch = new CountDownLatch(1);
        lastBestMove.set("");
        goTimestamp.set(0);

        send("ucinewgame");
        sleep(100);
        send(test.positionCommand());
        sleep(100);
        sendGo(test.moveTimeMs);

        boolean responded = bestMoveLatch.await(test.moveTimeMs + WAIT_BUFFER_MS, TimeUnit.MILLISECONDS);
        if (!responded) {
            record(test.description, false, "No bestmove within time limit");
            continue;
        }

        String bestMove = lastBestMove.get();
        long elapsed = bestMoveTimestamp.get() - goTimestamp.get();
        System.out.printf("  bestmove: %-8s  elapsed: %d ms%n", bestMove, elapsed);

        if (elapsed > test.moveTimeMs * 1.2)
            System.out.printf("  WARNING: %d ms exceeded movetime %d ms%n", elapsed, test.moveTimeMs);

        if (bestMove.equals("0000") || bestMove.isEmpty()) {
            record(test.description, false, "Engine returned null/zero bestmove");
        } else if (test.expectedMoves.length > 0 && !test.validates(bestMove)) {
            record(test.description, false,
                    "Expected one of " + Arrays.toString(test.expectedMoves) + " got " + bestMove);
        } else {
            record(test.description, true,
                    "bestmove=" + bestMove + " in " + elapsed + " ms"
                            + (test.expectedMoves.length > 0 ? " [validated]" : ""));
        }

        sleep(BETWEEN_TESTS_MS);
    }
    }

    // =========================================================================
    // Guess test runner
    // =========================================================================

    static void runGuessTests(List<GuessScenario> suite) throws Exception {
    header("GUESS-TABLE TESTS  (" + suite.size() + " scenarios)");
    IO.println("  What is tested:");
    IO.println("  1. Engine generates guess pairs while opponent thinks (guess[N] lines)");
    IO.println("  2. When opponent's actual move was guessed → 'Guess hit!' + hint used");
    IO.println("  3. When opponent's actual move was not guessed → 'Guess miss' + normal search");
    IO.println("  4. Response times are measured so you can compare hint vs non-hint speed");
    IO.println();

    for (int i = 0; i < suite.size(); i++) {
        GuessScenario test = suite.get(i);
        IO.println();
        subHeader((i + 1) + "/" + suite.size() + "  " + test.description);

        // Reset all guess counters for this scenario
        guessCount.set(0);
        sawGuessHit.set(false);
        sawGuessMiss.set(false);
        bestMoveLatch = new CountDownLatch(1);
        lastBestMove.set("");
        int engineLogSizeBefore = engineLog.size();

        send("ucinewgame");
        sleep(100);

        // ── Step 1: send first position and let engine search ────────────
        IO.println("  [Step 1] Sending first position and go...");
        send("position fen " + test.firstFen);
        sleep(100);
        sendGo(test.moveTimeMs);

        boolean step1ok = bestMoveLatch.await(test.moveTimeMs + WAIT_BUFFER_MS, TimeUnit.MILLISECONDS);
        if (!step1ok) {
            record(test.description, false, "Step 1: engine did not return bestmove");
            continue;
        }

        String move1 = lastBestMove.get();
        long elapsed1 = bestMoveTimestamp.get() - goTimestamp.get();
        System.out.printf("  [Step 1] bestmove=%-8s  elapsed=%d ms%n", move1, elapsed1);

        // ── Step 2: wait for the guessing thread to generate pairs ───────
        System.out.printf("  [Step 2] Waiting %d ms for guessing thread to run...%n", test.waitMs);
        sleep(test.waitMs);

        int guessesGenerated = guessCount.get();
        System.out.printf("  [Step 2] Guess pairs generated so far: %d%n", guessesGenerated);

        // Print a table of what was guessed (parse from engine log)
        printGuessTable(engineLog, engineLogSizeBefore);

        // Validate: if wait was long enough, at least 1 guess should exist
        boolean guessesExpected = test.waitMs >= 500;
        if (guessesExpected && guessesGenerated == 0) {
            IO.println("  WARNING: No guess pairs generated despite " + test.waitMs
                    + " ms wait — NN may be too slow or table is not being populated.");
        }

        // ── Step 3: send the opponent's actual reply and search ──────────
        IO.println("  [Step 3] Sending opponent's actual reply and go...");
        bestMoveLatch = new CountDownLatch(1);
        lastBestMove.set("");
        sawGuessHit.set(false);
        sawGuessMiss.set(false);

    // Build the actual second position from the engine's real first move
        Position afterOurMove = Position.fromFEN(test.firstFen).makeMove(Move.fromUci(move1));
        Position afterOpponentReply = afterOurMove.makeMove(Move.fromUci(test.opponentMoveUci()));

        send("position fen " + toFen(afterOpponentReply));
        sleep(100);
        sendGo(test.moveTimeMs);

        boolean step3ok = bestMoveLatch.await(test.moveTimeMs + WAIT_BUFFER_MS, TimeUnit.MILLISECONDS);
        if (!step3ok) {
            record(test.description, false, "Step 3: engine did not return bestmove after opponent reply");
            continue;
        }

        String move2 = lastBestMove.get();
        long elapsed2 = bestMoveTimestamp.get() - goTimestamp.get();
        System.out.printf("  [Step 3] bestmove=%-8s  elapsed=%d ms%n", move2, elapsed2);

        boolean hitDetected = sawGuessHit.get();
        boolean missDetected = sawGuessMiss.get();
        System.out.printf("  Guess hit: %b  |  Guess miss: %b%n", hitDetected, missDetected);

        // Consistency check: exactly one of hit or miss should fire
        if (hitDetected && missDetected) {
            IO.println("  WARNING: Both 'Guess hit' and 'Guess miss' were logged — inconsistent state.");
        } else if (!hitDetected && !missDetected && guessesGenerated > 0) {
            IO.println("  WARNING: Guesses were generated but neither hit nor miss was logged.");
            IO.println("           Check that the engine calls detectOpponentMove() correctly.");
        }

        // Pass/fail logic:
        //   If we expected a hit → engine must have logged "Guess hit" AND returned a valid move
        //   If we expected a miss → engine must have logged "Guess miss" AND returned a valid move
        //   If guesses were generated → engine must have logged one of hit or miss
        boolean validMove = !move2.isEmpty() && !move2.equals("0000");
        boolean outcomeMatchesExpectation;
        if (test.expectHit) {
            outcomeMatchesExpectation = hitDetected;
        } else {
            outcomeMatchesExpectation = missDetected || !guessesExpected;
        }

        boolean guessSystemWorking = guessesGenerated > 0 || !guessesExpected;

        boolean passed = validMove && guessSystemWorking &&
                (test.expectHit ? hitDetected : (missDetected || !guessesExpected));

        record(test.description, passed,
                "guesses=" + guessesGenerated
                        + " hit=" + hitDetected
                        + " miss=" + missDetected
                        + " expectedHit=" + test.expectHit
                        + " outcomeOk=" + outcomeMatchesExpectation
                        + " move=" + move2
                        + " elapsed=" + elapsed2 + "ms");

        sleep(BETWEEN_TESTS_MS);
    }
    }

    // Prints all guess[N] lines from the engine log since logStartIndex.
    private static void printGuessTable(List<String> log, int logStartIndex) {
    List<String> guessLines = new ArrayList<>();
    synchronized (Collections.unmodifiableList(log)) {
        for (int i = logStartIndex; i < log.size(); i++) {
            String l = log.get(i);
            if (l.contains("guess[")) guessLines.add(l);
        }
    }
    if (guessLines.isEmpty()) {
        IO.println("  (No guess pairs logged yet)");
        return;
    }
    IO.println("  Guess table contents:");
    for (String gl : guessLines) {
        // Strip "info string " prefix if present
        String display = gl.replaceFirst("^.*info string\\s*", "").trim();
        IO.println("    " + display);
    }
    }

    // =========================================================================
    // Perft test runner
    // =========================================================================

    static void runPerftTests(List<PerftExpected> suite) throws Exception {
    header("PERFT TESTS — Move Generation Correctness  (" + suite.size() + " cases)");
    IO.println("  Note: expected node counts assume ENABLE_EN_PASSANT = false.\n");

    for (PerftExpected test : suite) {
        subHeader(test.description + "  (depth " + test.depth + ")");

        awaitedPerftDepth = test.depth;
        perftLatch = new CountDownLatch(1);
        lastPerftNodes.set(-1);

        send("ucinewgame");
        sleep(100);
        send("position fen " + test.fen);
        sleep(100);

        long start = System.currentTimeMillis();
        send("perft " + test.depth);

        boolean received = perftLatch.await(30, TimeUnit.SECONDS);
        long elapsed = System.currentTimeMillis() - start;

        if (!received) {
            record(test.description, false, "Perft timed out");
            continue;
        }

        long actualNodes = lastPerftNodes.get();

        if (actualNodes < 0) {
            record(test.description, false, "Could not parse node count");
            continue;
        }

        boolean correct = actualNodes == test.expectedNodes;
        System.out.printf("  Depth %d  expected=%d  actual=%d  elapsed=%d ms%n",
                test.depth, test.expectedNodes, actualNodes, elapsed);
        record(test.description, correct,
                correct ? "node count matches" : "MISMATCH: got " + actualNodes
                                                 + " expected " + test.expectedNodes);

        sleep(BETWEEN_TESTS_MS);
    }
    }

    // =========================================================================
    // Stress test — rapid-fire positions to check stability
    // =========================================================================

    static void runStressTest() throws Exception {
    header("STRESS TEST — Rapid Search Stability");

    String[] fens = {
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/8/8/8/3k4/8/3P4/3K4 w - - 0 1",
            "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "rnbqkb1r/pp1p1ppp/2p2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4",
    };

    int passed = 0, failed = 0;
    long totalTime = 0;

    for (int i = 0; i < fens.length; i++) {
        int moveTime = 1500;
        bestMoveLatch = new CountDownLatch(1);
        lastBestMove.set("");

        send("ucinewgame");
        sleep(100);
        send("position fen " + fens[i]);
        sleep(50);
        sendGo(moveTime);

        boolean ok = bestMoveLatch.await(moveTime + WAIT_BUFFER_MS, TimeUnit.MILLISECONDS);
        long elapsed = ok ? bestMoveTimestamp.get() - goTimestamp.get() : -1;
        String move = lastBestMove.get();
        totalTime += Math.max(elapsed, 0);

        if (!ok || move.isEmpty() || move.equals("0000")) {
            System.out.printf("  [%d/%d] FAIL  move=%-8s  time=%d ms%n", i + 1, fens.length, move, elapsed);
            failed++;
        } else {
            System.out.printf("  [%d/%d] OK    move=%-8s  time=%d ms%n", i + 1, fens.length, move, elapsed);
            passed++;
        }
        sleep(200);
    }

    System.out.printf("%n  Stress: %d/%d passed, avg response %.0f ms%n",
            passed, fens.length, (double) totalTime / fens.length);
    record("Stress test (" + fens.length + " positions)", failed == 0,
            passed + "/" + fens.length + " valid, avg " + (totalTime / fens.length) + " ms");
    }

    // =========================================================================
    // Communication helpers
    // =========================================================================

    static void send(String command) throws IOException {
    IO.println("  CLIENT> " + command);
    writer.write(command);
    writer.newLine();
    writer.flush();
    }

    static void sendGo(int moveTimeMs) throws IOException {
    goTimestamp.set(System.currentTimeMillis());
    send("go movetime " + moveTimeMs);
    }

    // =========================================================================
    // FEN builders
    // =========================================================================

    /** Ranks from rank 8 → rank 1, white to move, full castling rights. */
    static String buildFen(String... ranks) {
    return String.join("/", ranks) + " w KQkq - 0 1";
    }

    // =========================================================================
    // Result recording and summary
    // =========================================================================

    static void record(String name, boolean passed, String detail) {
    results.add(new TestResult(name, passed, detail));
    IO.println("  " + (passed ? "✓ PASS" : "✗ FAIL") + "  " + detail);
    }

    static void printSummary() {
    IO.println();
    header("TEST SUMMARY");
    int pass = 0, fail = 0;
    for (TestResult r : results) {
        System.out.printf("  %-6s  %-55s  %s%n",
                r.passed ? "PASS" : "FAIL", r.name, r.detail);
        if (r.passed) pass++;
        else fail++;
    }
    System.out.printf("%n  Total: %d passed, %d failed, %d total%n", pass, fail, results.size());
    IO.println(fail == 0 ? "\n  All tests passed." : "\n  Some tests failed.");
    }

    static void header(String title) {
    IO.println();
    IO.println("=".repeat(70));
    IO.println("  " + title);
    IO.println("=".repeat(70));
    }

    static void subHeader(String title) {
    IO.println("  ── " + title);
    }

    static String toFen(Position pos) {
    StringBuilder sb = new StringBuilder();

    for (int rank = 7; rank >= 0; rank--) {
    int empty = 0;
    for (int file = 0; file < 8; file++) {
        int sq = rank * 8 + file;
        char piece = pos.getPieceAt(sq);
        if (piece == '.') {
            empty++;
        } else {
            if (empty > 0) {
                sb.append(empty);
                empty = 0;
            }
            sb.append(piece);
        }
    }
    if (empty > 0) sb.append(empty);
    if (rank > 0) sb.append('/');
    }

    sb.append(pos.whiteToMove ? " w " : " b ");

    StringBuilder castle = new StringBuilder();
    if (pos.whiteKingSideCastle) castle.append('K');
    if (pos.whiteQueenSideCastle) castle.append('Q');
    if (pos.blackKingSideCastle) castle.append('k');
    if (pos.blackQueenSideCastle) castle.append('q');
    sb.append(castle.isEmpty() ? "-" : castle.toString());

    sb.append(" - 0 1");
    return sb.toString();
}


