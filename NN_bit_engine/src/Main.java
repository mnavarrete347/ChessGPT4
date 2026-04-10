import java.io.*;
import java.util.*;
import ai.onnxruntime.*;

/**
 * UCI chess engine combining a bitboard position representation with a
 * negamax alpha-beta search and an optional ONNX neural-network move hint.
 *
 * <p>Legal move generation covers standard moves, castling, and promotion.
 * En-passant is disabled by default (see {@link #ENABLE_EN_PASSANT}).
 */
public class Main {

    // -------------------------------------------------------------------------
    // Engine-wide feature flags
    // -------------------------------------------------------------------------

    /** Set to {@code true} to enable en-passant capture generation. */
    public static final boolean ENABLE_EN_PASSANT = false;

    // -------------------------------------------------------------------------
    // ONNX model paths
    // -------------------------------------------------------------------------

    private static final String MODEL_PATH    = "models/chess_model_5050.onnx";
    private static final String MOVE_MAP_PATH = "models/move_map_5050.ser";

    // -------------------------------------------------------------------------
    // Search parameters (mutable at runtime via UCI "go" command)
    // -------------------------------------------------------------------------

    /** Maximum search depth. Reset to this value before each "go" command. */
    static int   maxDepth    = 100;
    static long  startTime;
    static long  timeLimit;
    /** Default per-move time budget in milliseconds. */
    static long  moveTimeMs  = 10_000;
    /** Score of the last completed root search iteration. */
    static int   lastScore   = 0;

    // -------------------------------------------------------------------------
    // Tapered-evaluation phase weights
    // -------------------------------------------------------------------------

    static final int KNIGHT_PHASE = 1;
    static final int BISHOP_PHASE = 1;
    static final int ROOK_PHASE   = 2;
    static final int QUEEN_PHASE  = 4;
    /** Total phase at the start of the game (2×(2×N + 2×B + 2×R + Q) = 24). */
    static final int MAX_PHASE    = 24;

    // -------------------------------------------------------------------------
    // Piece material values – midgame / endgame
    // -------------------------------------------------------------------------

    static final int MG_PAWN   = 100;
    static final int MG_KNIGHT = 320;
    static final int MG_BISHOP = 330;
    static final int MG_ROOK   = 500;
    static final int MG_QUEEN  = 900;
    static final int MG_KING   = 0;

    static final int EG_PAWN   = 130;
    static final int EG_KNIGHT = 320;
    static final int EG_BISHOP = 350;
    static final int EG_ROOK   = 500;
    static final int EG_QUEEN  = 950;
    static final int EG_KING   = 0;

    // -------------------------------------------------------------------------
    // Slider direction offsets
    // -------------------------------------------------------------------------

    static final int[] ROOK_OFFSETS   = {8, -8, 1, -1};
    static final int[] BISHOP_OFFSETS = {7, -7, 9, -9};
    static final int[] QUEEN_OFFSETS  = {8, -8, 1, -1, 7, -7, 9, -9};

    // -------------------------------------------------------------------------
    // Bitboard rank / file masks
    // -------------------------------------------------------------------------

    static final long FILE_A = 0x0101010101010101L;
    static final long FILE_H = 0x8080808080808080L;
    static final long RANK_1 = 0x00000000000000FFL;
    static final long RANK_3 = 0x0000000000FF0000L;
    static final long RANK_6 = 0x0000FF0000000000L;
    static final long RANK_8 = 0xFF00000000000000L;

    // -------------------------------------------------------------------------
    // Precomputed attack tables for knight and king
    // -------------------------------------------------------------------------

    static final long[] KNIGHT_ATTACKS = new long[64];
    static final long[] KING_ATTACKS   = new long[64];

    static {
        // Delta arrays for the 8 knight L-shapes and 8 king adjacencies.
        int[] knightDeltaRank = {-2, -2, -1, -1,  1,  1,  2,  2};
        int[] knightDeltaFile = {-1,  1, -2,  2, -2,  2, -1,  1};

        for (int sq = 0; sq < 64; sq++) {
            int rank = sq / 8;
            int file = sq % 8;

            long knightMask = 0L;
            for (int d = 0; d < 8; d++) {
                int nr = rank + knightDeltaRank[d];
                int nf = file + knightDeltaFile[d];
                if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                    knightMask |= 1L << (nr * 8 + nf);
                }
            }
            KNIGHT_ATTACKS[sq] = knightMask;

            long kingMask = 0L;
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) continue;
                    int nr = rank + dr;
                    int nf = file + df;
                    if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                        kingMask |= 1L << (nr * 8 + nf);
                    }
                }
            }
            KING_ATTACKS[sq] = kingMask;
        }
    }

    // -------------------------------------------------------------------------
    // Piece-square tables  (white's perspective; mirror with sq^56 for black)
    // -------------------------------------------------------------------------

    // --- Midgame PSTs ---
    static final int[] PAWN_PST = {
             0,  0,  0,  0,  0,  0,  0,  0, // Rank 1
             5, 10, 10,-20,-20, 10, 10,  5, // Rank 2
             5, -5,-10,  0,  0,-10, -5,  5, // Rank 3
             0,  0,  0, 20, 20,  0,  0,  0, // Rank 4
             5,  5, 10, 25, 25, 10,  5,  5, // Rank 5
            10, 10, 20, 30, 30, 20, 10, 10, // Rank 6
            50, 50, 50, 50, 50, 50, 50, 50, // Rank 7
             0,  0,  0,  0,  0,  0,  0,  0  // Rank 8
    };
    static final int[] KNIGHT_PST = {
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
    };
    static final int[] BISHOP_PST = {
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
    };
    static final int[] ROOK_PST = {
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             0,  0,  5, 10, 10,  5,  0,  0
    };
    static final int[] QUEEN_PST = {
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -10,  5,  5,  5,  5,  5,  0,-10,
              0,  0,  5,  5,  5,  5,  0, -5,
             -5,  0,  5,  5,  5,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
    };
    static final int[] KING_PST = {
             20, 30, 10,  0,  0, 10, 30, 20,
             20, 20,  0,  0,  0,  0, 20, 20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30
    };

    // --- Endgame PSTs ---
    static final int[] PAWN_PST_EG = {
             0,  0,  0,  0,  0,  0,  0,  0,
            -5, -5, -5, -5, -5, -5, -5, -5,
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 20, 20, 10, 10,  5,
            10, 20, 30, 40, 40, 30, 20, 10,
            30, 40, 50, 60, 60, 50, 40, 30,
            80, 80, 80, 80, 80, 80, 80, 80,
             0,  0,  0,  0,  0,  0,  0,  0
    };
    static final int[] KNIGHT_PST_EG = {
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
    };
    static final int[] BISHOP_PST_EG = {
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
    };
    static final int[] ROOK_PST_EG = {
             0,  0,  0,  5,  5,  0,  0,  0,
             0,  0,  0, 10, 10,  0,  0,  0,
             0,  0,  0, 10, 10,  0,  0,  0,
             0,  0,  0, 10, 10,  0,  0,  0,
             0,  0,  0, 10, 10,  0,  0,  0,
             0,  0,  0, 10, 10,  0,  0,  0,
            20, 20, 20, 20, 20, 20, 20, 20,
             0,  0,  0,  0,  0,  0,  0,  0
    };
    static final int[] QUEEN_PST_EG = {
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5, 10, 10,  5,  0, -5,
             -5,  0,  5, 10, 10,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
    };
    static final int[] KING_PST_EG = {
            -50,-30,-30,-30,-30,-30,-30,-50,
            -30,-10,  0,  0,  0,  0,-10,-30,
            -30,  0, 20, 30, 30, 20,  0,-30,
            -30,  0, 30, 40, 40, 30,  0,-30,
            -30,  0, 30, 40, 40, 30,  0,-30,
            -30,  0, 20, 30, 30, 20,  0,-30,
            -30,-10,  0,  0,  0,  0,-10,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
    };

    // =========================================================================
    // Entry point – UCI protocol loop
    // =========================================================================

    /**
     * Reads UCI commands from stdin and writes responses to stdout.
     * Loads the ONNX neural network on startup; falls back gracefully if
     * the model files are unavailable.
     */
    public static void main(String[] args) throws Exception {
        BufferedReader in  = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter    out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);

        Position     pos      = Position.startPos();
        NeuralEngine nnEngine = tryLoadNeuralEngine();

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
                pos = Position.startPos();

            } else if (line.startsWith("position")) {
                pos = parsePosition(line, pos);
                pos.printBoard();

            } else if (line.startsWith("go")) {
                parseGo(line);
                int bestMove = iterativeSearch(pos, nnEngine);
                out.println(bestMove == 0 ? "bestmove 0000" : "bestmove " + Move.toUci(bestMove));
                pos = pos.makeMove(bestMove);
                pos.printBoard();

            } else if (line.startsWith("perft")) {
                for (int depth = 1; depth <= 6; depth++) {
                    runPerft(pos, depth);
                }

            } else if (line.equals("quit")) {
                break;
            }
            // Other UCI commands (setoption, etc.) are silently ignored.
        }
        out.flush();
    }

    /** Attempts to load the neural engine; returns {@code null} on failure. */
    private static NeuralEngine tryLoadNeuralEngine() {
        try {
            Map<String, Integer> moveMap = loadMoveMap(MOVE_MAP_PATH);
            NeuralEngine engine = new NeuralEngine(MODEL_PATH, moveMap);
            System.out.println("Info string: Neural Engine loaded successfully.");
            return engine;
        } catch (Exception e) {
            System.out.println("Info string: Neural Engine failed: " + e.getMessage());
            return null; // Engine continues in pure bitboard mode.
        }
    }

    // =========================================================================
    // Perft (move-generation correctness / performance test)
    // =========================================================================

    public static void runPerft(Position pos, int depth) {
        long start = System.nanoTime();
        long nodes = perft(pos, depth);
        long end   = System.nanoTime();

        double seconds = (end - start) / 1_000_000_000.0;
        long   nps     = (long) (nodes / seconds);
        System.out.printf("Depth %d: %d nodes in %.3f s (NPS: %d)%n", depth, nodes, seconds, nps);
    }

    private static long perft(Position pos, int depth) {
        if (depth == 0) return 1;
        long nodes = 0;
        MoveList moves = pos.legalMoves();
        for (int i = 0; i < moves.count; i++) {
            nodes += perft(pos.makeMove(moves.moves[i]), depth - 1);
        }
        return nodes;
    }

    // =========================================================================
    // UCI command parsing helpers
    // =========================================================================

    /**
     * Parses a UCI {@code position} command and returns the resulting board state.
     *
     * <p>Supported formats:
     * <pre>
     *   position startpos [moves ...]
     *   position fen &lt;fen&gt; [moves ...]
     * </pre>
     */
    private static Position parsePosition(String cmd, Position currentPos) {
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
            i++;
            for (; i < tokens.length; i++) {
                pos = pos.makeMove(Move.fromUci(tokens[i]));
            }
        }
        return pos;
    }

    /**
     * Parses a UCI {@code go} command and updates the global search parameters.
     *
     * <p>Recognised tokens: {@code movetime <ms>}, {@code depth <n>}.
     * Unrecognised tokens reset parameters to their defaults.
     */
    static void parseGo(String cmd) {
        String[] tokens = cmd.trim().split("\\s+");
        // Reset to defaults before applying any overrides.
        moveTimeMs = 10_000;
        maxDepth   = 100;

        for (int i = 1; i < tokens.length; i++) {
            switch (tokens[i].toLowerCase()) {
                case "movetime":
                    if (i + 1 < tokens.length) { moveTimeMs = Long.parseLong(tokens[++i]); }
                    break;
                case "depth":
                    if (i + 1 < tokens.length) { maxDepth = Integer.parseInt(tokens[++i]); }
                    break;
                default:
                    break;
            }
        }
    }

    // =========================================================================
    // Search – iterative deepening, root, and negamax
    // =========================================================================

    /**
     * Runs iterative-deepening search and returns the best move found within
     * the configured time/depth budget.
     */
    static int iterativeSearch(Position pos, NeuralEngine nn) {
        startTime = System.currentTimeMillis();
        timeLimit = (long) (moveTimeMs * 0.9);

        // Query the neural network once at the root (skip if too little time).
        int rootNnMove = 0;
        if (moveTimeMs >= 800 && nn != null) {
            try {
                rootNnMove = nn.predictBestMove(pos, pos.legalMoves());
            } catch (Exception e) {
                System.out.println("Info string: NN error: " + e.getMessage());
            }
        }

        int bestMove = 0;
        for (int depth = 1; depth <= maxDepth; depth++) {
            int candidate = findBest(pos, depth, bestMove, rootNnMove);
            if (isTimeUp()) break;
            bestMove = candidate;
            if (Math.abs(lastScore) > 90_000) break; // Forced mate found.
        }
        return bestMove;
    }

    /**
     * Root negamax call: orders moves with NN/PV hints and returns the best move
     * found at the given {@code depth}.
     */
    static int findBest(Position pos, int depth, int prevBest, int nnHint) {
        MoveList moves = pos.legalMoves();
        if (moves.count == 0) return 0;

        orderMoves(pos, moves, prevBest, nnHint);

        int bestMove = moves.moves[0];
        int alpha    = -1_000_000;
        int beta     =  1_000_000;

        for (int i = 0; i < moves.count; i++) {
            Position next  = pos.makeMove(moves.moves[i]);
            // NN/PV hints only apply at the root; pass 0 for deeper plies.
            int score = -negamax(next, depth - 1, -beta, -alpha);

            if (score > alpha) {
                alpha    = score;
                bestMove = moves.moves[i];
                lastScore = score;
            }
            if (isTimeUp()) break;
        }
        return bestMove;
    }

    /**
     * Recursive negamax with alpha-beta pruning.
     * Uses pseudo-legal generation with a legality check after each move.
     */
    static int negamax(Position pos, int depth, int alpha, int beta) {
        if (isTimeUp() || depth == 0) return evaluate(pos);

        MoveList moves    = pos.pseudoLegalMoves();
        int      legals   = 0;

        orderMoves(pos, moves, 0, 0); // MVV-LVA ordering; no NN hint at depth.

        for (int i = 0; i < moves.count; i++) {
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue; // Illegal: own king in check.
            legals++;

            int score = -negamax(next, depth - 1, -beta, -alpha);
            if (score >= beta) return beta;  // Beta cutoff.
            if (score > alpha) alpha = score;
        }

        // Terminal node: checkmate or stalemate.
        if (legals == 0) {
            return pos.inCheck(pos.whiteToMove) ? (-100_000 + (maxDepth - depth)) : 0;
        }
        return alpha;
    }

    // =========================================================================
    // Evaluation
    // =========================================================================

    /**
     * Returns a tapered evaluation score relative to the side to move.
     * Interpolates linearly between the incrementally maintained midgame
     * and endgame scores stored in {@code pos}.
     */
    public static int evaluate(Position pos) {
        int phase      = Math.max(0, Math.min(pos.phase, MAX_PHASE));
        int totalScore = (pos.mgScore * phase + pos.egScore * (MAX_PHASE - phase)) / MAX_PHASE;
        return pos.whiteToMove ? totalScore : -totalScore;
    }

    // =========================================================================
    // Move ordering
    // =========================================================================

    /**
     * Scores and reorders {@code list} in place using selection sort.
     *
     * <p>Priority (highest first):
     * <ol>
     *   <li>Neural-network hint move</li>
     *   <li>PV move from the previous iteration</li>
     *   <li>Captures ordered by MVV-LVA; promotions</li>
     * </ol>
     */
    static void orderMoves(Position pos, MoveList list, int prevBest, int nnHint) {
        int[] scores = new int[list.count];

        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];
            if      (m == nnHint)   scores[i] = 20_000;
            else if (m == prevBest) scores[i] = 10_000;
            else                    scores[i] = scoreMove(pos, m);
        }

        // Simple selection sort – acceptable for typical move counts (~40).
        for (int i = 0; i < list.count - 1; i++) {
            for (int j = i + 1; j < list.count; j++) {
                if (scores[j] > scores[i]) {
                    int tmp = list.moves[i]; list.moves[i] = list.moves[j]; list.moves[j] = tmp;
                    int ts  = scores[i];    scores[i]    = scores[j];    scores[j]    = ts;
                }
            }
        }
    }

    /**
     * Returns a heuristic score for {@code move} used by move ordering.
     * Captures are scored by MVV-LVA; promotions receive a flat bonus.
     */
    static int scoreMove(Position pos, int move) {
        int  score    = 0;
        int  from     = Move.getFrom(move);
        int  to       = Move.getTo(move);
        long toBit    = 1L << to;

        if ((pos.allPieces & toBit) != 0) {
            int victim   = pieceValueForOrdering(pos.getPieceAt(to));
            int attacker = pieceValueForOrdering(pos.getPieceAt(from));
            score = 1000 + victim - (attacker / 10); // MVV-LVA
        }
        if (Move.getPromo(move) != 0) score += 900;
        return score;
    }

    /** Returns the MG material value of piece {@code p} (case-insensitive). */
    private static int pieceValueForOrdering(char p) {
        return switch (Character.toUpperCase(p)) {
            case 'P' -> MG_PAWN;
            case 'N' -> MG_KNIGHT;
            case 'B' -> MG_BISHOP;
            case 'R' -> MG_ROOK;
            case 'Q' -> MG_QUEEN;
            case 'K' -> 10_000; // High king value prevents "king takes" from ranking low.
            default  -> 0;
        };
    }

    // =========================================================================
    // Utility
    // =========================================================================

    private static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    /**
     * Deserialises and returns the move-index map from disk.
     *
     * @param path serialised {@code Map<String, Integer>} file path
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Integer> loadMoveMap(String path) throws Exception {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (Map<String, Integer>) ois.readObject();
        }
    }

    // =========================================================================
    // NeuralEngine – ONNX-based move prediction
    // =========================================================================

    /**
     * Wraps an ONNX Runtime session and produces a legal-move hint by
     * running the board through the network and selecting the highest-logit
     * move that is also present in the legal move list.
     */
    public static class NeuralEngine {

        private final OrtEnvironment env;
        private final OrtSession     session;
        private final String         inputName;

        /**
         * Array-based index → UCI string lookup (faster than a HashMap in
         * the per-node prediction loop).
         */
        private final String[] indexToMoveUci;

        public NeuralEngine(String modelPath, Map<String, Integer> moveMap) throws OrtException {
            this.env       = OrtEnvironment.getEnvironment();
            this.session   = env.createSession(modelPath, new OrtSession.SessionOptions());
            this.inputName = session.getInputNames().iterator().next();

            int maxIdx = moveMap.values().stream().mapToInt(Integer::intValue).max().orElse(0);
            this.indexToMoveUci = new String[maxIdx + 1];
            for (Map.Entry<String, Integer> e : moveMap.entrySet()) {
                indexToMoveUci[e.getValue()] = e.getKey();
            }
        }

        /**
         * Converts the board into a [1][13][8][8] float tensor.
         *
         * <p>Channels 0–11 encode each piece type's bitboard; channel 12
         * marks the "to" squares of all legal moves.
         */
        private float[][][][] boardToTensor(Position pos, MoveList legalMoves) {
            float[][][][] tensor = new float[1][13][8][8];

            long[] bitboards = {
                pos.wp, pos.wn, pos.wb, pos.wr, pos.wq, pos.wk,
                pos.bp, pos.bn, pos.bb, pos.br, pos.bq, pos.bk
            };
            for (int ch = 0; ch < 12; ch++) {
                long bb = bitboards[ch];
                while (bb != 0) {
                    int sq = Long.numberOfTrailingZeros(bb);
                    tensor[0][ch][sq / 8][sq % 8] = 1.0f;
                    bb &= bb - 1;
                }
            }
            // Channel 12: legal destination squares.
            for (int i = 0; i < legalMoves.count; i++) {
                int to = Move.getTo(legalMoves.moves[i]);
                tensor[0][12][to / 8][to % 8] = 1.0f;
            }
            return tensor;
        }

        /**
         * Runs inference and returns the highest-scoring legal move, or
         * {@code 0} if no legal match is found in the output distribution.
         */
        public int predictBestMove(Position pos, MoveList legalMoves) throws OrtException {
            float[][][][] inputData = boardToTensor(pos, legalMoves);

            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {

                float[] logits = ((float[][]) result.get(0).getValue())[0];

                // Build an index array sorted by descending logit (argmax order).
                Integer[] indices = new Integer[logits.length];
                for (int i = 0; i < logits.length; i++) indices[i] = i;
                Arrays.sort(indices, (a, b) -> Float.compare(logits[b], logits[a]));

                // Walk indices from best to worst; return the first legal move found.
                for (int idx : indices) {
                    String moveUci = indexToMoveUci[idx];
                    if (moveUci == null) continue;
                    for (int i = 0; i < legalMoves.count; i++) {
                        if (Move.toUci(legalMoves.moves[i]).equals(moveUci)) {
                            return legalMoves.moves[i];
                        }
                    }
                }
            }
            return 0; // No legal match found.
        }
    }

    // =========================================================================
    // Move encoding  (packed int: bits 0-5 = from, 6-11 = to, 12-14 = promo)
    // =========================================================================

    /**
     * Represents a chess move as a packed {@code int}.
     *
     * <pre>
     *   bits  0– 5 : from square (0–63)
     *   bits  6–11 : to square   (0–63)
     *   bits 12–14 : promotion   (0=none, 1=Q, 2=R, 3=B, 4=N)
     * </pre>
     */
    static final class Move {

        private Move() {} // Non-instantiable utility class.

        static int create(int from, int to, int promo) {
            return from | (to << 6) | (promo << 12);
        }

        static int  getFrom (int move) { return  move        & 0x3F; }
        static int  getTo   (int move) { return (move >> 6)  & 0x3F; }
        static int  getPromo(int move) { return (move >> 12) & 0x07; }

        /** Parses a UCI move string (e.g. {@code "e2e4"} or {@code "a7a8q"}). */
        static int fromUci(String s) {
            if (s == null || s.length() < 4) return 0;
            int from  = squareIndex(s.substring(0, 2));
            int to    = squareIndex(s.substring(2, 4));
            int promo = 0;
            if (s.length() >= 5) {
                promo = switch (s.charAt(4)) {
                    case 'q' -> 1;
                    case 'r' -> 2;
                    case 'b' -> 3;
                    case 'n' -> 4;
                    default  -> 0;
                };
            }
            return create(from, to, promo);
        }

        /** Converts a packed move to its UCI string representation. */
        static String toUci(int move) {
            String s = indexToSquare(getFrom(move)) + indexToSquare(getTo(move));
            return switch (getPromo(move)) {
                case 1  -> s + "q";
                case 2  -> s + "r";
                case 3  -> s + "b";
                case 4  -> s + "n";
                default -> s;
            };
        }

        /** Converts a UCI square name (e.g. {@code "e4"}) to a 0-based index. */
        static int squareIndex(String sq) {
            return (sq.charAt(1) - '1') * 8 + (sq.charAt(0) - 'a');
        }

        /** Converts a 0-based square index to a UCI square name. */
        static String indexToSquare(int idx) {
            return "" + (char) ('a' + idx % 8) + (char) ('1' + idx / 8);
        }
    }

    // =========================================================================
    // MoveList – fixed-capacity move buffer (avoids heap allocation per node)
    // =========================================================================

    /**
     * A fixed-capacity list of packed move ints.
     * 256 slots comfortably exceeds the theoretical maximum of 218 legal moves
     * in any reachable position.
     */
    static final class MoveList {
        final int[] moves = new int[256];
        int count = 0;

        void add(int move) { moves[count++] = move; }
    }

    // =========================================================================
    // Position – immutable board snapshot with incremental score bookkeeping
    // =========================================================================

    /**
     * Encodes a full chess position as twelve piece bitboards plus game-state
     * flags. Midgame/endgame scores and the game phase are maintained
     * incrementally: {@link #makeMove(int)} updates them in O(1) without
     * recomputing from scratch.
     *
     * <p>The class is effectively immutable: every {@code makeMove} call
     * returns a new {@code Position} and leaves {@code this} untouched.
     */
    static final class Position {

        // --- Piece bitboards ---
        final long wp, wn, wb, wr, wq, wk; // White pieces
        final long bp, bn, bb, br, bq, bk; // Black pieces

        // --- Aggregate occupancy ---
        final long whitePieces, blackPieces, allPieces;

        // --- Game state ---
        final boolean whiteToMove;
        final boolean whiteKingSideCastle, whiteQueenSideCastle;
        final boolean blackKingSideCastle, blackQueenSideCastle;
        final int enPassantSq; // -1 if no en-passant target square.

        // --- Incremental evaluation scores ---
        int mgScore; // White-relative midgame material + PST sum.
        int egScore; // White-relative endgame material + PST sum.
        int phase;   // Remaining phase material (decreases as pieces come off).

        // -----------------------------------------------------------------------
        // Construction
        // -----------------------------------------------------------------------

        Position(long[] pieces, boolean whiteToMove,
                 boolean wK, boolean wQ, boolean bK, boolean bQ, int epSq) {
            this.wp = pieces[0]; this.wn = pieces[1]; this.wb = pieces[2];
            this.wr = pieces[3]; this.wq = pieces[4]; this.wk = pieces[5];
            this.bp = pieces[6]; this.bn = pieces[7]; this.bb = pieces[8];
            this.br = pieces[9]; this.bq = pieces[10]; this.bk = pieces[11];

            this.whitePieces = wp | wn | wb | wr | wq | wk;
            this.blackPieces = bp | bn | bb | br | bq | bk;
            this.allPieces   = whitePieces | blackPieces;

            this.whiteToMove          = whiteToMove;
            this.whiteKingSideCastle  = wK;
            this.whiteQueenSideCastle = wQ;
            this.blackKingSideCastle  = bK;
            this.blackQueenSideCastle = bQ;
            this.enPassantSq          = epSq;
        }

        /** Returns the standard starting position. */
        static Position startPos() {
            return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }

        /** Parses a FEN string and returns the corresponding position. */
        static Position fromFEN(String fen) {
            String[] parts     = fen.trim().split("\\s+");
            String   placement = parts[0];
            boolean  wtm       = parts.length <= 1 || parts[1].equals("w");

            long[] pieces = new long[12];
            int rank = 7, file = 0;

            for (int i = 0; i < placement.length(); i++) {
                char c = placement.charAt(i);
                if (c == '/') {
                    rank--;
                    file = 0;
                } else if (Character.isDigit(c)) {
                    file += c - '0';
                } else {
                    int pIdx = getPieceIndex(c);
                    pieces[pIdx] |= 1L << (rank * 8 + file);
                    file++;
                }
            }

            boolean[] cr  = parseCastlingRights(parts.length > 2 ? parts[2] : "-");
            String    epT = parts.length > 3 ? parts[3] : "-";
            int       epSq = epT.equals("-") ? -1 : Move.squareIndex(epT);

            Position pos = new Position(pieces, wtm, cr[0], cr[1], cr[2], cr[3], epSq);
            pos.calculateInitialScores();
            return pos;
        }

        // -----------------------------------------------------------------------
        // Incremental score helpers
        // -----------------------------------------------------------------------

        /**
         * Computes {@link #mgScore}, {@link #egScore}, and {@link #phase}
         * from scratch. Called once during {@link #fromFEN}.
         */
        private void calculateInitialScores() {
            mgScore = 0;
            egScore = 0;
            phase   = 0;
            for (int i = 0; i < 12; i++) {
                long bb = bitboardByIndex(i);
                while (bb != 0) {
                    int sq = Long.numberOfTrailingZeros(bb);
                    mgScore += mgValue(i, sq);
                    egScore += egValue(i, sq);
                    phase   += piecePhase(i);
                    bb &= bb - 1;
                }
            }
        }

        /**
         * Returns the white-relative midgame material + PST value for piece
         * type {@code idx} on square {@code sq}.
         */
        private static int mgValue(int idx, int sq) {
            boolean isWhite  = idx < 6;
            int     tableIdx = isWhite ? sq : sq ^ 56;
            int score = switch (idx % 6) {
                case 0 -> MG_PAWN   + PAWN_PST[tableIdx];
                case 1 -> MG_KNIGHT + KNIGHT_PST[tableIdx];
                case 2 -> MG_BISHOP + BISHOP_PST[tableIdx];
                case 3 -> MG_ROOK   + ROOK_PST[tableIdx];
                case 4 -> MG_QUEEN  + QUEEN_PST[tableIdx];
                case 5 -> MG_KING   + KING_PST[tableIdx];
                default -> 0;
            };
            return isWhite ? score : -score;
        }

        /** Endgame counterpart of {@link #mgValue}. */
        private static int egValue(int idx, int sq) {
            boolean isWhite  = idx < 6;
            int     tableIdx = isWhite ? sq : sq ^ 56;
            int score = switch (idx % 6) {
                case 0 -> EG_PAWN   + PAWN_PST_EG[tableIdx];
                case 1 -> EG_KNIGHT + KNIGHT_PST_EG[tableIdx];
                case 2 -> EG_BISHOP + BISHOP_PST_EG[tableIdx];
                case 3 -> EG_ROOK   + ROOK_PST_EG[tableIdx];
                case 4 -> EG_QUEEN  + QUEEN_PST_EG[tableIdx];
                case 5 -> EG_KING   + KING_PST_EG[tableIdx];
                default -> 0;
            };
            return isWhite ? score : -score;
        }

        /** Returns the phase contribution of piece type {@code idx}. */
        private static int piecePhase(int idx) {
            return switch (idx % 6) {
                case 1 -> KNIGHT_PHASE;
                case 2 -> BISHOP_PHASE;
                case 3 -> ROOK_PHASE;
                case 4 -> QUEEN_PHASE;
                default -> 0; // Pawns and kings contribute nothing.
            };
        }

        // -----------------------------------------------------------------------
        // makeMove – the core of incremental update logic
        // -----------------------------------------------------------------------

        /**
         * Returns a new {@code Position} after applying {@code move}.
         * All score fields are updated incrementally (no full recompute).
         */
        Position makeMove(int move) {
            long[] next   = {wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk};
            int    from   = Move.getFrom(move);
            int    to     = Move.getTo(move);
            int    promo  = Move.getPromo(move);
            long   fromBit = 1L << from;
            long   toBit   = 1L << to;

            int nextMg    = mgScore;
            int nextEg    = egScore;
            int nextPhase = phase;

            // 1. Identify moving piece.
            int movingIdx = -1;
            for (int i = 0; i < 12; i++) {
                if ((next[i] & fromBit) != 0) { movingIdx = i; break; }
            }

            // 2. Deduct 'from' contribution.
            nextMg -= mgValue(movingIdx, from);
            nextEg -= egValue(movingIdx, from);

            // 3. Handle captures (clear captured piece from its bitboard).
            if ((allPieces & toBit) != 0) {
                for (int i = 0; i < 12; i++) {
                    if ((next[i] & toBit) != 0) {
                        nextMg    -= mgValue(i, to);
                        nextEg    -= egValue(i, to);
                        nextPhase -= piecePhase(i);
                        next[i]   ^= toBit;
                        break;
                    }
                }
            }

            // 4. En passant capture.
            int nextEpSq = -1;
            if (ENABLE_EN_PASSANT && (movingIdx == 0 || movingIdx == 6) && to == enPassantSq) {
                int capturedSq  = whiteToMove ? to - 8 : to + 8;
                int victimIdx   = whiteToMove ? 6 : 0;
                nextMg   -= mgValue(victimIdx, capturedSq);
                nextEg   -= egValue(victimIdx, capturedSq);
                next[victimIdx] ^= 1L << capturedSq;
            }

            // 5. Move piece to destination.
            next[movingIdx] ^= fromBit | toBit;
            nextMg += mgValue(movingIdx, to);
            nextEg += egValue(movingIdx, to);

            // 6. Promotion: replace pawn with chosen piece.
            if (promo != 0 && (movingIdx == 0 || movingIdx == 6)) {
                nextMg -= mgValue(movingIdx, to);
                nextEg -= egValue(movingIdx, to);
                next[movingIdx] ^= toBit;

                // Map promotion code to bitboard index (white queen=4, rook=3, bishop=2, knight=1;
                // black queen=10, rook=9, bishop=8, knight=7).
                int promoIdx = whiteToMove
                        ? (promo == 1 ? 4 : promo == 2 ? 3 : promo == 3 ? 2 : 1)
                        : (promo == 1 ? 10 : promo == 2 ? 9 : promo == 3 ? 8 : 7);

                next[promoIdx]  |= toBit;
                nextMg          += mgValue(promoIdx, to);
                nextEg          += egValue(promoIdx, to);
                nextPhase       += piecePhase(promoIdx);
            }

            // 7. Double pawn push → set en-passant target square.
            if ((movingIdx == 0 || movingIdx == 6) && Math.abs(to - from) == 16) {
                nextEpSq = (from + to) / 2;
            }

            // 8. Update castling rights.
            boolean wK = whiteKingSideCastle,  wQ = whiteQueenSideCastle;
            boolean bK = blackKingSideCastle,  bQ = blackQueenSideCastle;

            if (from == 4  || to == 4)  { wK = false; wQ = false; } // White king moved/captured
            if (from == 60 || to == 60) { bK = false; bQ = false; } // Black king moved/captured
            if (from == 0  || to == 0)  wQ = false;                 // a1 rook
            if (from == 7  || to == 7)  wK = false;                 // h1 rook
            if (from == 56 || to == 56) bQ = false;                 // a8 rook
            if (from == 63 || to == 63) bK = false;                 // h8 rook

            // 9. Move rooks during castling.
            if (movingIdx == 5 && from == 4) {        // White king castles
                if (to == 6) {                         // Kingside
                    nextMg  -= mgValue(3, 7);  nextEg -= egValue(3, 7);
                    nextMg  += mgValue(3, 5);  nextEg += egValue(3, 5);
                    next[3] ^= (1L << 7 | 1L << 5);
                } else if (to == 2) {                  // Queenside
                    nextMg  -= mgValue(3, 0);  nextEg -= egValue(3, 0);
                    nextMg  += mgValue(3, 3);  nextEg += egValue(3, 3);
                    next[3] ^= (1L << 0 | 1L << 3);
                }
            } else if (movingIdx == 11 && from == 60) { // Black king castles
                if (to == 62) {                          // Kingside
                    nextMg  -= mgValue(9, 63); nextEg -= egValue(9, 63);
                    nextMg  += mgValue(9, 61); nextEg += egValue(9, 61);
                    next[9] ^= (1L << 63 | 1L << 61);
                } else if (to == 58) {                   // Queenside
                    nextMg  -= mgValue(9, 56); nextEg -= egValue(9, 56);
                    nextMg  += mgValue(9, 59); nextEg += egValue(9, 59);
                    next[9] ^= (1L << 56 | 1L << 59);
                }
            }

            Position nextPos = new Position(next, !whiteToMove, wK, wQ, bK, bQ, nextEpSq);
            nextPos.mgScore  = nextMg;
            nextPos.egScore  = nextEg;
            nextPos.phase    = nextPhase;
            return nextPos;
        }

        // -----------------------------------------------------------------------
        // Move generation
        // -----------------------------------------------------------------------

        /** Returns all pseudo-legal moves (may leave own king in check). */
        MoveList pseudoLegalMoves() {
            MoveList list = new MoveList();
            boolean  w    = whiteToMove;
            genPawn          (list, w);
            genKnight        (list, w);
            genSlidingPieces (list, w, BISHOP_OFFSETS, w ? wb : bb);
            genSlidingPieces (list, w, ROOK_OFFSETS,   w ? wr : br);
            genSlidingPieces (list, w, QUEEN_OFFSETS,  w ? wq : bq);
            genKing          (list, w);
            return list;
        }

        /** Filters pseudo-legal moves and returns only fully legal ones. */
        MoveList legalMoves() {
            MoveList pseudo = pseudoLegalMoves();
            MoveList legal  = new MoveList();
            for (int i = 0; i < pseudo.count; i++) {
                Position next = makeMove(pseudo.moves[i]);
                // After the move, it is the opponent's turn; check if OUR king is now attacked.
                if (!next.inCheck(!next.whiteToMove)) {
                    legal.add(pseudo.moves[i]);
                }
            }
            return legal;
        }

        void genPawn(MoveList list, boolean isWhite) {
            long enemy = isWhite ? blackPieces : whitePieces;
            long empty = ~allPieces;

            if (isWhite) {
                long single = (wp << 8) & empty;
                addPawnMoves(list, single & ~RANK_8, 8,  false);
                addPawnMoves(list, single &  RANK_8, 8,  true);
                addPawnMoves(list, ((single & RANK_3) << 8) & empty, 16, false);

                long capL = (wp << 7) & enemy & ~FILE_H;
                addPawnMoves(list, capL & ~RANK_8, 7, false);
                addPawnMoves(list, capL &  RANK_8, 7, true);

                long capR = (wp << 9) & enemy & ~FILE_A;
                addPawnMoves(list, capR & ~RANK_8, 9, false);
                addPawnMoves(list, capR &  RANK_8, 9, true);

                if (enPassantSq != -1) {
                    long ep = 1L << enPassantSq;
                    if (((wp << 7) & ep & ~FILE_H) != 0) list.add(Move.create(enPassantSq - 7, enPassantSq, 0));
                    if (((wp << 9) & ep & ~FILE_A) != 0) list.add(Move.create(enPassantSq - 9, enPassantSq, 0));
                }
            } else {
                long single = (bp >> 8) & empty;
                addPawnMoves(list, single & ~RANK_1, -8,  false);
                addPawnMoves(list, single &  RANK_1, -8,  true);
                addPawnMoves(list, ((single & RANK_6) >> 8) & empty, -16, false);

                long capL = (bp >> 9) & enemy & ~FILE_H;
                addPawnMoves(list, capL & ~RANK_1, -9, false);
                addPawnMoves(list, capL &  RANK_1, -9, true);

                long capR = (bp >> 7) & enemy & ~FILE_A;
                addPawnMoves(list, capR & ~RANK_1, -7, false);
                addPawnMoves(list, capR &  RANK_1, -7, true);

                if (enPassantSq != -1) {
                    long ep = 1L << enPassantSq;
                    if (((bp >> 9) & ep & ~FILE_H) != 0) list.add(Move.create(enPassantSq + 9, enPassantSq, 0));
                    if (((bp >> 7) & ep & ~FILE_A) != 0) list.add(Move.create(enPassantSq + 7, enPassantSq, 0));
                }
            }
        }

        void genKnight(MoveList list, boolean isWhite) {
            long knights  = isWhite ? wn : bn;
            long friendly = isWhite ? whitePieces : blackPieces;
            while (knights != 0) {
                int from = Long.numberOfTrailingZeros(knights);
                serializeMoves(list, from, KNIGHT_ATTACKS[from] & ~friendly);
                knights &= knights - 1;
            }
        }

        void genKing(MoveList list, boolean isWhite) {
            int  from     = Long.numberOfTrailingZeros(isWhite ? wk : bk);
            long friendly = isWhite ? whitePieces : blackPieces;
            serializeMoves(list, from, KING_ATTACKS[from] & ~friendly);

            if (isWhite) {
                if (whiteKingSideCastle  && (allPieces & 0x60L) == 0
                        && !isSquareAttacked(4, false) && !isSquareAttacked(5, false) && !isSquareAttacked(6, false))
                    list.add(Move.create(4, 6, 0));
                if (whiteQueenSideCastle && (allPieces & 0x0EL) == 0
                        && !isSquareAttacked(4, false) && !isSquareAttacked(3, false) && !isSquareAttacked(2, false))
                    list.add(Move.create(4, 2, 0));
            } else {
                if (blackKingSideCastle  && (allPieces & 0x6000000000000000L) == 0
                        && !isSquareAttacked(60, true) && !isSquareAttacked(61, true) && !isSquareAttacked(62, true))
                    list.add(Move.create(60, 62, 0));
                if (blackQueenSideCastle && (allPieces & 0x0E00000000000000L) == 0
                        && !isSquareAttacked(60, true) && !isSquareAttacked(59, true) && !isSquareAttacked(58, true))
                    list.add(Move.create(60, 58, 0));
            }
        }

        void genSlidingPieces(MoveList list, boolean isWhite, int[] offsets, long pieces) {
            long friendly = isWhite ? whitePieces : blackPieces;
            long enemy    = isWhite ? blackPieces : whitePieces;

            while (pieces != 0) {
                int from = Long.numberOfTrailingZeros(pieces);
                for (int offset : offsets) {
                    int cur = from;
                    while (true) {
                        int next = cur + offset;
                        if (next < 0 || next >= 64 || isWrap(cur, next, offset)) break;
                        long bit = 1L << next;
                        if ((friendly & bit) != 0) break;        // Blocked by own piece.
                        list.add(Move.create(from, next, 0));
                        if ((enemy & bit) != 0) break;           // Captured; stop ray.
                        cur = next;
                    }
                }
                pieces &= pieces - 1;
            }
        }

        /** Serialises all set bits of {@code destinations} as moves from {@code from}. */
        private void serializeMoves(MoveList list, int from, long destinations) {
            while (destinations != 0) {
                int to = Long.numberOfTrailingZeros(destinations);
                list.add(Move.create(from, to, 0));
                destinations &= destinations - 1;
            }
        }

        /**
         * Serialises pawn moves; generates four promotion variants when
         * {@code isPromotion} is {@code true}.
         */
        private void addPawnMoves(MoveList list, long destinations, int offset, boolean isPromotion) {
            while (destinations != 0) {
                int to   = Long.numberOfTrailingZeros(destinations);
                int from = to - offset;
                if (isPromotion) {
                    list.add(Move.create(from, to, 1)); // Queen
                    list.add(Move.create(from, to, 4)); // Knight
                    list.add(Move.create(from, to, 2)); // Rook
                    list.add(Move.create(from, to, 3)); // Bishop
                } else {
                    list.add(Move.create(from, to, 0));
                }
                destinations &= destinations - 1;
            }
        }

        // -----------------------------------------------------------------------
        // Check / attack detection
        // -----------------------------------------------------------------------

        /** Returns {@code true} if the king of {@code isWhiteKing} is in check. */
        boolean inCheck(boolean isWhiteKing) {
            long kingBoard = isWhiteKing ? wk : bk;
            if (kingBoard == 0) return true; // No king → treat as in check.
            int kingSq = Long.numberOfTrailingZeros(kingBoard);
            return isSquareAttacked(kingSq, !isWhiteKing);
        }

        /** Returns {@code true} if {@code sq} is attacked by the side indicated by {@code byWhite}. */
        boolean isSquareAttacked(int sq, boolean byWhite) {
            long bit = 1L << sq;

            // Pawn attacks
            if (byWhite) {
                if (((bit >> 7) & wp & ~FILE_A) != 0) return true;
                if (((bit >> 9) & wp & ~FILE_H) != 0) return true;
            } else {
                if (((bit << 7) & bp & ~FILE_H) != 0) return true;
                if (((bit << 9) & bp & ~FILE_A) != 0) return true;
            }

            // Knight and king attacks (lookup tables)
            if ((KNIGHT_ATTACKS[sq] & (byWhite ? wn : bn)) != 0) return true;
            if ((KING_ATTACKS[sq]   & (byWhite ? wk : bk)) != 0) return true;

            // Slider attacks
            return isAttackedBySlider(sq, byWhite);
        }

        /**
         * Returns {@code true} if a rook/queen or bishop/queen belonging to
         * {@code byWhite} attacks {@code sq} along any ray.
         */
        private boolean isAttackedBySlider(int sq, boolean byWhite) {
            long straightAttackers = byWhite ? (wr | wq) : (br | bq);
            long diagonalAttackers = byWhite ? (wb | wq) : (bb | bq);

            for (int offset : ROOK_OFFSETS) {
                int cur = sq;
                while (true) {
                    int next = cur + offset;
                    if (next < 0 || next >= 64 || isWrap(cur, next, offset)) break;
                    long bit = 1L << next;
                    if ((straightAttackers & bit) != 0) return true;
                    if ((allPieces        & bit) != 0) break; // Ray blocked.
                    cur = next;
                }
            }
            for (int offset : BISHOP_OFFSETS) {
                int cur = sq;
                while (true) {
                    int next = cur + offset;
                    if (next < 0 || next >= 64 || isWrap(cur, next, offset)) break;
                    long bit = 1L << next;
                    if ((diagonalAttackers & bit) != 0) return true;
                    if ((allPieces         & bit) != 0) break;
                    cur = next;
                }
            }
            return false;
        }

        /**
         * Returns {@code true} when moving from {@code from} to {@code to}
         * would illegally wrap around the board edge.
         *
         * <p>Vertical offsets (±8) must not change the file; all other sliding
         * offsets must move by exactly one file per step.
         */
        private boolean isWrap(int from, int to, int offset) {
            int fileDiff = Math.abs((to & 7) - (from & 7));
            return Math.abs(offset) == 8 ? fileDiff != 0 : fileDiff != 1;
        }

        // -----------------------------------------------------------------------
        // Board display / queries
        // -----------------------------------------------------------------------

        /** Returns the character representing the piece on {@code sq}, or {@code '.'}. */
        char getPieceAt(int sq) {
            long bit = 1L << sq;
            if ((wp & bit) != 0) return 'P'; if ((wn & bit) != 0) return 'N';
            if ((wb & bit) != 0) return 'B'; if ((wr & bit) != 0) return 'R';
            if ((wq & bit) != 0) return 'Q'; if ((wk & bit) != 0) return 'K';
            if ((bp & bit) != 0) return 'p'; if ((bn & bit) != 0) return 'n';
            if ((bb & bit) != 0) return 'b'; if ((br & bit) != 0) return 'r';
            if ((bq & bit) != 0) return 'q'; if ((bk & bit) != 0) return 'k';
            return '.';
        }

        /** Prints the board to stdout with rank and file labels. */
        void printBoard() {
            System.out.println();
            for (int rank = 7; rank >= 0; rank--) {
                System.out.print((rank + 1) + "  ");
                for (int file = 0; file < 8; file++) {
                    System.out.print(getPieceAt(rank * 8 + file) + " ");
                }
                System.out.println();
            }
            System.out.println("\n   a b c d e f g h\n");
        }

        // -----------------------------------------------------------------------
        // FEN parsing helpers
        // -----------------------------------------------------------------------

        /** Maps FEN piece characters to 0-based bitboard array indices. */
        private static int getPieceIndex(char c) {
            return switch (c) {
                case 'P' -> 0; case 'N' -> 1; case 'B' -> 2;
                case 'R' -> 3; case 'Q' -> 4; case 'K' -> 5;
                case 'p' -> 6; case 'n' -> 7; case 'b' -> 8;
                case 'r' -> 9; case 'q' -> 10; case 'k' -> 11;
                default  -> -1;
            };
        }

        /** Parses the castling-rights field of a FEN string. */
        private static boolean[] parseCastlingRights(String s) {
            return new boolean[]{
                s.contains("K"), s.contains("Q"),
                s.contains("k"), s.contains("q")
            };
        }

        /** Returns the bitboard for piece index {@code i} (0 = wp … 11 = bk). */
        private long bitboardByIndex(int i) {
            return switch (i) {
                case  0 -> wp; case  1 -> wn; case  2 -> wb;
                case  3 -> wr; case  4 -> wq; case  5 -> wk;
                case  6 -> bp; case  7 -> bn; case  8 -> bb;
                case  9 -> br; case 10 -> bq; case 11 -> bk;
                default -> 0L;
            };
        }
    }
}
