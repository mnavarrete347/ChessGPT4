import java.io.*;
import java.util.*;
import ai.onnxruntime.*;

/**
 * UCI chess engine: hybrid MCTS + negamax alpha-beta search, guided by an
 * ONNX neural network that supplies both a policy (move prior probabilities)
 * and a value (position score) head.
 *
 * <h2>Architecture overview</h2>
 * <pre>
 *   UCI loop
 *     └─ mctsSearch()          – time-managed MCTS driver
 *          ├─ MctsNode          – tree node (visits, Q, policy prior)
 *          ├─ select()          – PUCT descent to a leaf
 *          ├─ expandNode()      – NeuralEngine.policyPriors() → child priors
 *          ├─ rollout()         – NeuralEngine.evaluatePosition() (value head)
 *          │                      OR shallow negamax if NN unavailable
 *          └─ backpropagate()   – propagate score up the tree
 *
 *   iterativeNegamax()         – complete fallback when NN is absent
 *   negamax()                  – recursive alpha-beta (also used as rollout)
 *   evaluate()                 – tapered PST evaluation (leaf scorer)
 * </pre>
 *
 * <h2>Neural network contract</h2>
 * The ONNX model must expose <em>two</em> output tensors (in index order):
 * <ol>
 *   <li><b>Policy logits</b> – shape [1, N_MOVES]: raw logits over the full
 *       move vocabulary. Filtered to legal moves and softmax-normalised here.</li>
 *   <li><b>Value</b> – shape [1, 1]: a scalar in (−1, +1) representing the
 *       expected outcome from the <em>current side's</em> perspective
 *       (+1 = win, 0 = draw, −1 = loss). Train with a tanh output activation.</li>
 * </ol>
 *
 * <p>Legal move generation covers standard moves, castling, and promotion.
 * En-passant is disabled by default ({@link #ENABLE_EN_PASSANT}).
 */
public class Main {

    // =========================================================================
    // Engine-wide flags and model paths
    // =========================================================================

    /** Set to {@code true} to enable en-passant capture generation. */
    public static final boolean ENABLE_EN_PASSANT = false;

    private static final String MODEL_PATH    = "models/chess_model_EVH_150200.onnx";
    private static final String MOVE_MAP_PATH = "models/move_map_EVH_150200.ser";

    // =========================================================================
    // MCTS hyper-parameters
    // =========================================================================

    /**
     * PUCT exploration constant C. Controls the trade-off between exploiting
     * high-prior moves and exploring low-visit moves.
     * Typical range: 1.0 – 2.5. Tune by self-play at fixed time controls.
     */
    static double EXPLORATION_C = 1.5;

    /**
     * Depth of the negamax rollout used when the NN value head is unavailable
     * or when {@link #USE_NN_VALUE} is {@code false}.
     */
    static int ROLLOUT_DEPTH = 3;

    /**
     * When {@code true} (default) the NN value head drives leaf evaluation.
     * Flip to {@code false} to substitute shallow negamax (useful for testing).
     */
    static boolean USE_NN_VALUE = true;

    // =========================================================================
    // Search parameters (overridden at runtime by the UCI "go" command)
    // =========================================================================

    /** Maximum negamax depth ceiling; used only in iterative-deepening fallback. */
    static int  maxDepth  = 100;
    static long startTime;
    static long timeLimit;
    /** Per-move time budget in milliseconds (default 10 s). */
    static long moveTimeMs = 10_000;
    /** Centipawn score of the last completed root search (for UCI info output). */
    static int  lastScore  = 0;

    // =========================================================================
    // Tapered-evaluation phase weights
    // =========================================================================

    static final int KNIGHT_PHASE = 1;
    static final int BISHOP_PHASE = 1;
    static final int ROOK_PHASE   = 2;
    static final int QUEEN_PHASE  = 4;
    /** Total phase material at game start: 2 × (2N + 2B + 2R + Q) per side = 24. */
    static final int MAX_PHASE    = 24;

    // =========================================================================
    // Material values – midgame and endgame
    // =========================================================================

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

    // =========================================================================
    // Slider direction offsets
    // =========================================================================

    static final int[] ROOK_OFFSETS   = {  8, -8,  1, -1 };
    static final int[] BISHOP_OFFSETS = {  7, -7,  9, -9 };
    static final int[] QUEEN_OFFSETS  = {  8, -8,  1, -1, 7, -7, 9, -9 };

    // =========================================================================
    // Bitboard masks
    // =========================================================================

    static final long FILE_A = 0x0101010101010101L;
    static final long FILE_H = 0x8080808080808080L;
    static final long RANK_1 = 0x00000000000000FFL;
    static final long RANK_3 = 0x0000000000FF0000L;
    static final long RANK_6 = 0x0000FF0000000000L;
    static final long RANK_8 = 0xFF00000000000000L;

    // =========================================================================
    // Precomputed attack tables
    // =========================================================================

    static final long[] KNIGHT_ATTACKS = new long[64];
    static final long[] KING_ATTACKS   = new long[64];

    static {
        int[] knightDR = { -2, -2, -1, -1,  1,  1,  2,  2 };
        int[] knightDF = { -1,  1, -2,  2, -2,  2, -1,  1 };

        for (int sq = 0; sq < 64; sq++) {
            int r = sq / 8, f = sq % 8;

            long kn = 0L;
            for (int d = 0; d < 8; d++) {
                int nr = r + knightDR[d], nf = f + knightDF[d];
                if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) kn |= 1L << (nr * 8 + nf);
            }
            KNIGHT_ATTACKS[sq] = kn;

            long ki = 0L;
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) continue;
                    int nr = r + dr, nf = f + df;
                    if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) ki |= 1L << (nr * 8 + nf);
                }
            }
            KING_ATTACKS[sq] = ki;
        }
    }

    // =========================================================================
    // Piece-square tables (white's perspective; black mirrors with sq ^ 56)
    // =========================================================================

    // --- Midgame PSTs ---
    static final int[] PAWN_PST = {
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10,-20,-20, 10, 10,  5,
             5, -5,-10,  0,  0,-10, -5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
             0,  0,  0,  0,  0,  0,  0,  0
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
            20, 20, 20, 20, 20, 20, 20, 20, // Seventh-rank bonus
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
     * Reads UCI commands from stdin and dispatches them.
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
                out.println("id name team_java");
                out.println("id author team_java_bryan");
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
                int bestMove = mctsSearch(pos, nnEngine);
                out.println(bestMove == 0 ? "bestmove 0000" : "bestmove " + Move.toUci(bestMove));
                pos = pos.makeMove(bestMove);
                pos.printBoard();

            } else if (line.startsWith("perft")) {
                for (int d = 1; d <= 6; d++) runPerft(pos, d);

            } else if (line.equals("quit")) {
                break;
            }
            // Other UCI commands (setoption, etc.) are silently ignored.
        }
        out.flush();
    }

    /** Attempts to load the neural engine; returns {@code null} on any failure. */
    private static NeuralEngine tryLoadNeuralEngine() {
        try {
            Map<String, Integer> moveMap = loadMoveMap(MOVE_MAP_PATH);
            NeuralEngine engine = new NeuralEngine(MODEL_PATH, moveMap);
            System.out.println("info string Neural Engine loaded successfully.");
            return engine;
        } catch (Exception e) {
            System.out.println("info string Neural Engine unavailable: " + e.getMessage());
            System.out.println("info string Falling back to pure negamax search.");
            return null;
        }
    }

    // =========================================================================
    // Perft – move-generation correctness / performance test
    // =========================================================================

    public static void runPerft(Position pos, int depth) {
        long start = System.nanoTime();
        long nodes = perft(pos, depth);
        long end   = System.nanoTime();
        double secs = (end - start) / 1_000_000_000.0;
        System.out.printf("Depth %d: %d nodes in %.3f s (NPS: %d)%n",
                depth, nodes, secs, (long)(nodes / secs));
    }

    private static long perft(Position pos, int depth) {
        if (depth == 0) return 1;
        long nodes = 0;
        MoveList moves = pos.legalMoves();
        for (int i = 0; i < moves.count; i++) nodes += perft(pos.makeMove(moves.moves[i]), depth - 1);
        return nodes;
    }

    // =========================================================================
    // UCI command parsers
    // =========================================================================

    /**
     * Parses a UCI {@code position} command and returns the resulting board.
     * Handles both {@code startpos} and {@code fen <...>} formats, with an
     * optional trailing {@code moves} token list.
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
            for (i++; i < tokens.length; i++) pos = pos.makeMove(Move.fromUci(tokens[i]));
        }
        return pos;
    }

    /**
     * Parses a UCI {@code go} command and updates {@link #moveTimeMs} /
     * {@link #maxDepth}. Defaults are restored first so omitted tokens revert
     * to a sane state.
     */
    static void parseGo(String cmd) {
        String[] tokens = cmd.trim().split("\\s+");
        moveTimeMs = 10_000;
        maxDepth   = 100;

        for (int i = 1; i < tokens.length; i++) {
            switch (tokens[i].toLowerCase()) {
                case "movetime" -> { if (i + 1 < tokens.length) moveTimeMs = Long.parseLong(tokens[++i]); }
                case "depth"    -> { if (i + 1 < tokens.length) maxDepth   = Integer.parseInt(tokens[++i]); }
                default         -> { /* ignore wtime/btime/movestogo/etc. */ }
            }
        }
    }

    // =========================================================================
    // MCTS – top-level search driver
    // =========================================================================

    /**
     * Runs Monte Carlo Tree Search for up to {@link #moveTimeMs} milliseconds
     * and returns the most-visited root child's move.
     *
     * <p>Graceful degradation:
     * <ul>
     *   <li>NN absent → iterative-deepening negamax (original behaviour).</li>
     *   <li>NN present but value head disabled → MCTS with negamax rollouts.</li>
     *   <li>NN fully enabled → MCTS with NN policy priors + value head.</li>
     * </ul>
     *
     * @param pos root position to search from
     * @param nn  neural engine, or {@code null} for pure negamax fallback
     * @return packed move int, or 0 if the position is already terminal
     */
    static int mctsSearch(Position pos, NeuralEngine nn) {
        startTime = System.currentTimeMillis();
        timeLimit = (long)(moveTimeMs * 0.95);

        // Degrade to iterative-deepening negamax when no NN is available.
        if (nn == null) return iterativeNegamax(pos, null);

        MoveList rootMoves = pos.legalMoves();
        if (rootMoves.count == 0) return 0;
        if (rootMoves.count == 1) return rootMoves.moves[0]; // Forced move.

        // Build and immediately expand the root node.
        MctsNode root = new MctsNode(pos, 0, null, 1.0f);
        try {
            expandNode(root, nn);
        } catch (Exception e) {
            System.out.println("info string MCTS root expand failed: " + e.getMessage());
            return iterativeNegamax(pos, nn);
        }

        // Simulation loop.
        int sims = 0;
        while (!isTimeUp()) {
            MctsNode leaf = select(root);

            double value;
            try {
                // Expand non-terminal leaves before evaluating.
                if (leaf.children == null && !leaf.isTerminal) expandNode(leaf, nn);
                value = rollout(leaf, nn);
            } catch (Exception e) {
                // NN mid-search error: fall back to PST score for this node only.
                value = Math.tanh(evaluate(leaf.pos) / 400.0);
            }

            backpropagate(leaf, value);
            sims++;
        }

        // Report search statistics to the UCI GUI.
        MctsNode best = mostVisitedChild(root);
        if (best == null) return rootMoves.moves[0];

        lastScore = (int)(best.q() * 400); // Convert (−1,+1) → centipawns.
        System.out.printf("info depth %d score cp %d nodes %d%n", sims, lastScore, sims);

        return best.moveFromParent;
    }

    // =========================================================================
    // MCTS – four core operations
    // =========================================================================

    /**
     * Selection: descend the tree by PUCT until an unexpanded or terminal leaf
     * is reached.
     *
     * <p>PUCT score for child {@code c} with parent {@code p}:
     * <pre>  Q(c) + C × P(c) × √N(p) / (1 + N(c))</pre>
     * <ul>
     *   <li>Q  – mean backed-up value (exploration target)</li>
     *   <li>P  – policy prior from the neural network</li>
     *   <li>N  – visit count</li>
     *   <li>C  – {@link #EXPLORATION_C}</li>
     * </ul>
     */
    private static MctsNode select(MctsNode node) {
        while (node.children != null && node.children.length > 0) {
            double   sqrtN    = Math.sqrt(node.visits);
            MctsNode bestChild = null;
            double   bestScore = Double.NEGATIVE_INFINITY;

            for (MctsNode child : node.children) {
                double q     = child.visits == 0 ? 0.0 : child.totalValue / child.visits;
                double u     = EXPLORATION_C * child.prior * sqrtN / (1.0 + child.visits);
                double score = q + u;
                if (score > bestScore) { bestScore = score; bestChild = child; }
            }
            if (bestChild == null) break;
            node = bestChild;
        }
        return node;
    }

    /**
     * Expansion: generate all legal moves from {@code node}, query the neural
     * network for softmax policy priors, and create child nodes.
     *
     * <p>If the position has no legal moves the node is flagged as terminal
     * and no children are created.
     */
    private static void expandNode(MctsNode node, NeuralEngine nn) throws OrtException {
        MoveList legal = node.pos.legalMoves();
        if (legal.count == 0) {
            node.isTerminal = true;
            return;
        }

        float[]   priors   = nn.policyPriors(node.pos, legal);
        node.children = new MctsNode[legal.count];
        for (int i = 0; i < legal.count; i++) {
            node.children[i] = new MctsNode(
                    node.pos.makeMove(legal.moves[i]),
                    legal.moves[i],
                    node,
                    priors[i]);
        }
    }

    /**
     * Rollout: evaluates a leaf node and returns a score in (−1, +1) from the
     * perspective of the side to move at that node.
     *
     * <ul>
     *   <li>Terminal → ±1 (checkmate) or 0 (stalemate).</li>
     *   <li>NN enabled → value head output (tanh-normalised by the model).</li>
     *   <li>NN disabled → shallow negamax score mapped via tanh.</li>
     * </ul>
     */
    private static double rollout(MctsNode node, NeuralEngine nn) throws OrtException {
        if (node.isTerminal) {
            return node.pos.inCheck(node.pos.whiteToMove) ? -1.0 : 0.0;
        }
        if (nn != null && USE_NN_VALUE) {
            return nn.evaluatePosition(node.pos);
        }
        // Negamax fallback: map centipawn score to (−1, +1).
        return Math.tanh(negamax(node.pos, ROLLOUT_DEPTH, -1_000_000, 1_000_000) / 400.0);
    }

    /**
     * Backpropagation: walk from {@code node} to the root, incrementing visit
     * counts and accumulating values. The value is negated at each level
     * because parent and child represent opposite sides.
     */
    private static void backpropagate(MctsNode node, double value) {
        double v = value;
        while (node != null) {
            node.visits++;
            node.totalValue += v;
            v    = -v;        // Flip perspective as we ascend.
            node = node.parent;
        }
    }

    /**
     * Returns the child of {@code root} with the highest visit count (the
     * standard MCTS move-selection rule), or {@code null} if there are none.
     */
    private static MctsNode mostVisitedChild(MctsNode root) {
        if (root.children == null || root.children.length == 0) return null;
        MctsNode best = root.children[0];
        for (MctsNode c : root.children) if (c.visits > best.visits) best = c;
        return best;
    }

    // =========================================================================
    // MctsNode – MCTS tree node
    // =========================================================================

    /**
     * A single node in the MCTS search tree.
     *
     * <p>Fields:
     * <ul>
     *   <li>{@link #pos}            – board position at this node</li>
     *   <li>{@link #moveFromParent} – packed move that produced this position</li>
     *   <li>{@link #prior}          – P(move | parent), from the NN policy head</li>
     *   <li>{@link #parent}         – reference to parent node (null at root)</li>
     *   <li>{@link #visits}         – simulation visit count N</li>
     *   <li>{@link #totalValue}     – sum of backed-up values (Q = total / visits)</li>
     *   <li>{@link #children}       – child nodes; null until expanded</li>
     *   <li>{@link #isTerminal}     – true when no legal moves exist</li>
     * </ul>
     */
    static final class MctsNode {
        final Position pos;
        final int      moveFromParent;
        final float    prior;
        final MctsNode parent;

        int        visits     = 0;
        double     totalValue = 0.0;
        MctsNode[] children   = null;
        boolean    isTerminal = false;

        MctsNode(Position pos, int moveFromParent, MctsNode parent, float prior) {
            this.pos            = pos;
            this.moveFromParent = moveFromParent;
            this.parent         = parent;
            this.prior          = prior;
        }

        /** Mean backed-up value Q from this node's side-to-move perspective. */
        double q() { return visits == 0 ? 0.0 : totalValue / visits; }
    }

    // =========================================================================
    // Negamax – fallback search and MCTS rollout evaluator
    // =========================================================================

    /**
     * Iterative-deepening negamax: the complete search fallback used when the
     * NN is unavailable. Preserves the original engine behaviour exactly.
     *
     * <p>When {@code nn} is non-null its policy head is used for a single
     * move-ordering hint at the root (the one-shot approach from the original
     * engine).
     */
    static int iterativeNegamax(Position pos, NeuralEngine nn) {
        startTime = System.currentTimeMillis();
        timeLimit = (long)(moveTimeMs * 0.9);

        int rootNnMove = 0;
        if (moveTimeMs >= 800 && nn != null) {
            try {
                rootNnMove = nn.topPolicyMove(pos, pos.legalMoves());
            } catch (Exception e) {
                System.out.println("info string NN hint error: " + e.getMessage());
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
     * Root negamax call: orders moves with PV and NN hints and returns the
     * best move found at {@code depth}.
     */
    static int findBest(Position pos, int depth, int prevBest, int nnHint) {
        MoveList moves = pos.legalMoves();
        if (moves.count == 0) return 0;

        orderMoves(pos, moves, prevBest, nnHint);

        int bestMove = moves.moves[0];
        int alpha    = -1_000_000;
        int beta     =  1_000_000;

        for (int i = 0; i < moves.count; i++) {
            int score = -negamax(pos.makeMove(moves.moves[i]), depth - 1, -beta, -alpha);
            if (score > alpha) {
                alpha     = score;
                bestMove  = moves.moves[i];
                lastScore = score;
            }
            if (isTimeUp()) break;
        }
        return bestMove;
    }

    /**
     * Recursive negamax with alpha-beta pruning.
     * Uses pseudo-legal generation with a post-move legality check.
     */
    static int negamax(Position pos, int depth, int alpha, int beta) {
        if (isTimeUp() || depth == 0) return evaluate(pos);

        MoveList moves  = pos.pseudoLegalMoves();
        int      legals = 0;
        orderMoves(pos, moves, 0, 0);

        for (int i = 0; i < moves.count; i++) {
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue; // Illegal: leaves own king in check.
            legals++;

            int score = -negamax(next, depth - 1, -beta, -alpha);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        if (legals == 0) {
            // Checkmate: penalise by remaining depth so faster mates rank higher.
            return pos.inCheck(pos.whiteToMove) ? (-100_000 + (maxDepth - depth)) : 0;
        }
        return alpha;
    }

    // =========================================================================
    // Static evaluation – tapered PST
    // =========================================================================

    /**
     * Returns a tapered evaluation score relative to the side to move.
     * Linearly interpolates between the incrementally maintained midgame and
     * endgame scores stored in {@code pos}.
     */
    public static int evaluate(Position pos) {
        int phase = Math.max(0, Math.min(pos.phase, MAX_PHASE));
        int score = (pos.mgScore * phase + pos.egScore * (MAX_PHASE - phase)) / MAX_PHASE;
        return pos.whiteToMove ? score : -score;
    }

    // =========================================================================
    // Move ordering
    // =========================================================================

    /**
     * Scores and reorders {@code list} in-place via selection sort.
     *
     * <p>Priority (highest first):
     * <ol>
     *   <li>NN policy-head hint (20 000)</li>
     *   <li>PV move from the previous iteration (10 000)</li>
     *   <li>Captures ordered by MVV-LVA; promotions (+900)</li>
     * </ol>
     */
    static void orderMoves(Position pos, MoveList list, int prevBest, int nnHint) {
        int[] scores = new int[list.count];
        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];
            if      (m == nnHint)   scores[i] = 20_000;
            else if (m == prevBest) scores[i] = 10_000;
            else                    scores[i] = scoreMoveForOrdering(pos, m);
        }
        // Selection sort – acceptable for the typical ~40 legal moves per node.
        for (int i = 0; i < list.count - 1; i++) {
            for (int j = i + 1; j < list.count; j++) {
                if (scores[j] > scores[i]) {
                    int t = list.moves[i]; list.moves[i] = list.moves[j]; list.moves[j] = t;
                    int s = scores[i];     scores[i]     = scores[j];     scores[j]     = s;
                }
            }
        }
    }

    /** MVV-LVA capture score plus a flat promotion bonus. */
    static int scoreMoveForOrdering(Position pos, int move) {
        int  score = 0;
        long toBit = 1L << Move.getTo(move);
        if ((pos.allPieces & toBit) != 0) {
            int victim   = pieceValueForOrdering(pos.getPieceAt(Move.getTo(move)));
            int attacker = pieceValueForOrdering(pos.getPieceAt(Move.getFrom(move)));
            score = 1000 + victim - (attacker / 10);
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
            case 'K' -> 10_000; // High value keeps "king takes" from ranking near the bottom.
            default  -> 0;
        };
    }

    // =========================================================================
    // Utility
    // =========================================================================

    private static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Integer> loadMoveMap(String path) throws Exception {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (Map<String, Integer>) ois.readObject();
        }
    }

    // =========================================================================
    // NeuralEngine – ONNX policy + value inference
    // =========================================================================

    /**
     * Wraps an ONNX Runtime session exposing two output heads.
     *
     * <h3>Input tensor</h3>
     * Shape [1, 13, 8, 8]:
     * <ul>
     *   <li>Channels 0–11: one binary plane per piece type (wp → ch0, … bk → ch11).</li>
     *   <li>Channel 12: legal-move destination squares (binary mask over all legal moves).</li>
     * </ul>
     *
     * <h3>Output tensor 0 – policy logits</h3>
     * Shape [1, N_MOVES]. Raw logits over the full move vocabulary.
     * {@link #policyPriors} masks to legal moves and applies softmax.
     *
     * <h3>Output tensor 1 – value</h3>
     * Shape [1, 1]. Expected game outcome in (−1, +1) from the current side's
     * perspective. {@link #evaluatePosition} returns this scalar directly; the
     * model must use a tanh output activation to match this contract.
     */
    public static class NeuralEngine {

        private final OrtEnvironment env;
        private final OrtSession     session;
        private final String         inputName;

        /**
         * Array-based index → UCI string lookup for the full move vocabulary.
         * Faster than a HashMap in the per-node hot path.
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

        // ── Shared inference kernel ───────────────────────────────────────────

        /**
         * Builds the [1, 13, 8, 8] input tensor and runs the ONNX session.
         * The caller owns the returned {@link OrtSession.Result} and must close it.
         */
        private OrtSession.Result runInference(Position pos, MoveList legal) throws OrtException {
            float[][][][] input = new float[1][13][8][8];

            // Channels 0-11: piece bitplanes.
            long[] bbs = { pos.wp, pos.wn, pos.wb, pos.wr, pos.wq, pos.wk,
                           pos.bp, pos.bn, pos.bb, pos.br, pos.bq, pos.bk };
            for (int ch = 0; ch < 12; ch++) {
                long bb = bbs[ch];
                while (bb != 0) {
                    int sq = Long.numberOfTrailingZeros(bb);
                    input[0][ch][sq / 8][sq % 8] = 1.0f;
                    bb &= bb - 1;
                }
            }
            // Channel 12: legal-move destination mask.
            for (int i = 0; i < legal.count; i++) {
                int to = Move.getTo(legal.moves[i]);
                input[0][12][to / 8][to % 8] = 1.0f;
            }

            try (OnnxTensor tensor = OnnxTensor.createTensor(env, input)) {
                return session.run(Collections.singletonMap(inputName, tensor));
            }
        }

        // ── Policy head ───────────────────────────────────────────────────────

        /**
         * Returns softmax-normalised policy priors over {@code legal}, aligned
         * to the same index order as {@code legal.moves[0..count-1]}.
         *
         * <p>Moves not present in the model's vocabulary receive a logit of
         * −∞ before softmax (effectively probability 0), after which the
         * distribution is renormalised. If all legal moves are unknown a
         * uniform prior is returned so that search can continue.
         */
        float[] policyPriors(Position pos, MoveList legal) throws OrtException {
            try (OrtSession.Result result = runInference(pos, legal)) {
                float[] logits = ((float[][]) result.get(0).getValue())[0];
                return softmaxOverLegal(logits, legal);
            }
        }

        /**
         * Returns the single highest-logit legal move — used by the negamax
         * fallback for move-ordering hints at the root. Returns 0 if no legal
         * move is found in the model's vocabulary.
         */
        int topPolicyMove(Position pos, MoveList legal) throws OrtException {
            try (OrtSession.Result result = runInference(pos, legal)) {
                float[] logits = ((float[][]) result.get(0).getValue())[0];

                // Sort indices by descending logit.
                Integer[] idx = new Integer[logits.length];
                for (int i = 0; i < logits.length; i++) idx[i] = i;
                Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));

                for (int i : idx) {
                    String uci = indexToMoveUci[i];
                    if (uci == null) continue;
                    for (int j = 0; j < legal.count; j++) {
                        if (Move.toUci(legal.moves[j]).equals(uci)) return legal.moves[j];
                    }
                }
                return 0;
            }
        }

        /**
         * Masks raw logits to the legal move set and applies numerically stable
         * softmax. Returns a probability array aligned to {@code legal}.
         */
        private float[] softmaxOverLegal(float[] logits, MoveList legal) {
            // Gather logits for each legal move; use -∞ for unknown moves.
            float[] legalLogits = new float[legal.count];
            for (int i = 0; i < legal.count; i++) {
                String uci = Move.toUci(legal.moves[i]);
                legalLogits[i] = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < indexToMoveUci.length; j++) {
                    if (uci.equals(indexToMoveUci[j])) { legalLogits[i] = logits[j]; break; }
                }
            }

            // Subtract max for numerical stability before exp().
            float max = Float.NEGATIVE_INFINITY;
            for (float v : legalLogits) if (v > max) max = v;

            float[] priors = new float[legal.count];
            float   sum    = 0f;
            for (int i = 0; i < legal.count; i++) {
                priors[i] = (float) Math.exp(legalLogits[i] - max);
                sum       += priors[i];
            }

            if (sum > 0f) {
                for (int i = 0; i < priors.length; i++) priors[i] /= sum;
            } else {
                // All legal moves unknown to the model: use a uniform prior.
                Arrays.fill(priors, 1.0f / legal.count);
            }
            return priors;
        }

        // ── Value head ────────────────────────────────────────────────────────

        /**
         * Queries the NN value head and returns the expected outcome for the
         * side to move at {@code pos}.
         *
         * <p>The model's second output tensor (index 1) must have shape [1, 1]
         * and a tanh output activation so that values are already in (−1, +1).
         * No additional normalisation is applied here.
         *
         * @param pos position to evaluate
         * @return scalar in (−1, +1); +1 means the side to move is winning
         */
        double evaluatePosition(Position pos) throws OrtException {
            // Channel 12 requires legal moves; compute them once.
            MoveList legal = pos.legalMoves();
            try (OrtSession.Result result = runInference(pos, legal)) {
                // Output index 1 is the value head: shape [1, 1].
                float[][] value = (float[][]) result.get(1).getValue();
                return value[0][0];
            }
        }
    }

    // =========================================================================
    // Move encoding  (packed int: bits 0-5 = from, 6-11 = to, 12-14 = promo)
    // =========================================================================

    /**
     * Encodes a chess move as a packed {@code int}.
     *
     * <pre>
     *   bits  0– 5 : from square (0–63)
     *   bits  6–11 : to square   (0–63)
     *   bits 12–14 : promotion   (0=none, 1=Q, 2=R, 3=B, 4=N)
     * </pre>
     */
    static final class Move {

        private Move() {}

        static int    create  (int from, int to, int promo) { return from | (to << 6) | (promo << 12); }
        static int    getFrom (int move) { return  move        & 0x3F; }
        static int    getTo   (int move) { return (move >> 6)  & 0x3F; }
        static int    getPromo(int move) { return (move >> 12) & 0x07; }

        /** Parses a UCI move string such as {@code "e2e4"} or {@code "a7a8q"}. */
        static int fromUci(String s) {
            if (s == null || s.length() < 4) return 0;
            int from  = squareIndex(s.substring(0, 2));
            int to    = squareIndex(s.substring(2, 4));
            int promo = s.length() >= 5 ? switch (s.charAt(4)) {
                case 'q' -> 1; case 'r' -> 2; case 'b' -> 3; case 'n' -> 4; default -> 0;
            } : 0;
            return create(from, to, promo);
        }

        /** Converts a packed move to its UCI string representation. */
        static String toUci(int move) {
            String base = indexToSquare(getFrom(move)) + indexToSquare(getTo(move));
            return switch (getPromo(move)) {
                case 1 -> base + "q"; case 2 -> base + "r";
                case 3 -> base + "b"; case 4 -> base + "n";
                default -> base;
            };
        }

        static int    squareIndex   (String sq) { return (sq.charAt(1) - '1') * 8 + (sq.charAt(0) - 'a'); }
        static String indexToSquare (int idx)   { return "" + (char)('a' + idx % 8) + (char)('1' + idx / 8); }
    }

    // =========================================================================
    // MoveList – fixed-capacity move buffer
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
     * incrementally so that {@link #makeMove(int)} runs in O(1).
     *
     * <p>The class is effectively immutable: every {@link #makeMove(int)} call
     * returns a fresh {@code Position} and leaves {@code this} untouched.
     */
    static final class Position {

        // --- Piece bitboards (one bit per occupied square) ---
        final long wp, wn, wb, wr, wq, wk; // White pieces
        final long bp, bn, bb, br, bq, bk; // Black pieces

        // --- Aggregate occupancy ---
        final long whitePieces, blackPieces, allPieces;

        // --- Game state ---
        final boolean whiteToMove;
        final boolean whiteKingSideCastle, whiteQueenSideCastle;
        final boolean blackKingSideCastle, blackQueenSideCastle;
        /** En-passant target square, or −1 if unavailable. */
        final int enPassantSq;

        // --- Incremental evaluation (white-relative) ---
        int mgScore; // Midgame material + PST sum
        int egScore; // Endgame material + PST sum
        int phase;   // Remaining phase material (decreases as pieces leave)

        // ── Construction ──────────────────────────────────────────────────────

        Position(long[] p, boolean wtm,
                 boolean wK, boolean wQ, boolean bK, boolean bQ, int epSq) {
            wp = p[0]; wn = p[1]; wb = p[2]; wr = p[3]; wq = p[4]; wk = p[5];
            bp = p[6]; bn = p[7]; bb = p[8]; br = p[9]; bq = p[10]; bk = p[11];

            whitePieces = wp | wn | wb | wr | wq | wk;
            blackPieces = bp | bn | bb | br | bq | bk;
            allPieces   = whitePieces | blackPieces;

            whiteToMove          = wtm;
            whiteKingSideCastle  = wK;
            whiteQueenSideCastle = wQ;
            blackKingSideCastle  = bK;
            blackQueenSideCastle = bQ;
            enPassantSq          = epSq;
        }

        static Position startPos() {
            return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }

        static Position fromFEN(String fen) {
            String[] parts = fen.trim().split("\\s+");
            boolean  wtm   = parts.length <= 1 || parts[1].equals("w");
            long[]   p     = new long[12];
            int      rank  = 7, file = 0;

            for (char c : parts[0].toCharArray()) {
                if      (c == '/')                  { rank--; file = 0; }
                else if (Character.isDigit(c))      { file += c - '0'; }
                else {
                    int idx = getPieceIndex(c);
                    if (idx >= 0) p[idx] |= 1L << (rank * 8 + file);
                    file++;
                }
            }

            boolean[] cr  = parseCastlingRights(parts.length > 2 ? parts[2] : "-");
            String    ept = parts.length > 3 ? parts[3] : "-";
            int       epSq = ept.equals("-") ? -1 : Move.squareIndex(ept);

            Position pos = new Position(p, wtm, cr[0], cr[1], cr[2], cr[3], epSq);
            pos.calculateInitialScores();
            return pos;
        }

        // ── Incremental scoring ───────────────────────────────────────────────

        /**
         * Computes {@link #mgScore}, {@link #egScore}, and {@link #phase} from
         * scratch. Called once by {@link #fromFEN}; thereafter scores are kept
         * current by {@link #makeMove}.
         */
        private void calculateInitialScores() {
            mgScore = 0; egScore = 0; phase = 0;
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
         * White-relative midgame material + PST value for piece type {@code idx}
         * on square {@code sq}. Negative for black pieces.
         */
        private static int mgValue(int idx, int sq) {
            boolean w = idx < 6;
            int     t = w ? sq : sq ^ 56; // Mirror vertically for black.
            int s = switch (idx % 6) {
                case 0 -> MG_PAWN   + PAWN_PST[t];
                case 1 -> MG_KNIGHT + KNIGHT_PST[t];
                case 2 -> MG_BISHOP + BISHOP_PST[t];
                case 3 -> MG_ROOK   + ROOK_PST[t];
                case 4 -> MG_QUEEN  + QUEEN_PST[t];
                case 5 -> MG_KING   + KING_PST[t];
                default -> 0;
            };
            return w ? s : -s;
        }

        /** Endgame counterpart of {@link #mgValue}. */
        private static int egValue(int idx, int sq) {
            boolean w = idx < 6;
            int     t = w ? sq : sq ^ 56;
            int s = switch (idx % 6) {
                case 0 -> EG_PAWN   + PAWN_PST_EG[t];
                case 1 -> EG_KNIGHT + KNIGHT_PST_EG[t];
                case 2 -> EG_BISHOP + BISHOP_PST_EG[t];
                case 3 -> EG_ROOK   + ROOK_PST_EG[t];
                case 4 -> EG_QUEEN  + QUEEN_PST_EG[t];
                case 5 -> EG_KING   + KING_PST_EG[t];
                default -> 0;
            };
            return w ? s : -s;
        }

        /** Phase contribution of piece type {@code idx} (0 for pawns and kings). */
        private static int piecePhase(int idx) {
            return switch (idx % 6) {
                case 1 -> KNIGHT_PHASE;
                case 2 -> BISHOP_PHASE;
                case 3 -> ROOK_PHASE;
                case 4 -> QUEEN_PHASE;
                default -> 0;
            };
        }

        // ── makeMove ─────────────────────────────────────────────────────────

        /**
         * Returns a new {@code Position} after applying {@code move}.
         * All scores are updated incrementally; no full recompute is done.
         */
        Position makeMove(int move) {
            long[] next  = { wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk };
            int    from  = Move.getFrom(move), to = Move.getTo(move), promo = Move.getPromo(move);
            long   fBit  = 1L << from, tBit = 1L << to;
            int    nMg   = mgScore, nEg = egScore, nPh = phase;

            // 1. Identify the moving piece.
            int movIdx = -1;
            for (int i = 0; i < 12; i++) if ((next[i] & fBit) != 0) { movIdx = i; break; }

            // 2. Deduct the 'from' square's PST contribution.
            nMg -= mgValue(movIdx, from);
            nEg -= egValue(movIdx, from);

            // 3. Handle captures: remove the victim from its bitboard.
            if ((allPieces & tBit) != 0) {
                for (int i = 0; i < 12; i++) {
                    if ((next[i] & tBit) != 0) {
                        nMg   -= mgValue(i, to);
                        nEg   -= egValue(i, to);
                        nPh   -= piecePhase(i);
                        next[i] ^= tBit;
                        break;
                    }
                }
            }

            // 4. En passant capture: remove the captured pawn from an adjacent square.
            int nextEpSq = -1;
            if (ENABLE_EN_PASSANT && (movIdx == 0 || movIdx == 6) && to == enPassantSq) {
                int capSq  = whiteToMove ? to - 8 : to + 8;
                int vicIdx = whiteToMove ? 6 : 0;
                nMg -= mgValue(vicIdx, capSq);
                nEg -= egValue(vicIdx, capSq);
                next[vicIdx] ^= 1L << capSq;
            }

            // 5. Place the moving piece on the destination square.
            next[movIdx] ^= fBit | tBit;
            nMg += mgValue(movIdx, to);
            nEg += egValue(movIdx, to);

            // 6. Promotion: swap pawn for the chosen piece type.
            if (promo != 0 && (movIdx == 0 || movIdx == 6)) {
                nMg -= mgValue(movIdx, to);
                nEg -= egValue(movIdx, to);
                next[movIdx] ^= tBit;
                // Map promotion code → bitboard index.
                // White: Q=4, R=3, B=2, N=1. Black: Q=10, R=9, B=8, N=7.
                int pIdx = whiteToMove
                        ? (promo == 1 ? 4 : promo == 2 ? 3 : promo == 3 ? 2 : 1)
                        : (promo == 1 ? 10 : promo == 2 ? 9 : promo == 3 ? 8 : 7);
                next[pIdx] |= tBit;
                nMg += mgValue(pIdx, to);
                nEg += egValue(pIdx, to);
                nPh += piecePhase(pIdx);
            }

            // 7. Double pawn push: set the en-passant target square.
            if ((movIdx == 0 || movIdx == 6) && Math.abs(to - from) == 16) {
                nextEpSq = (from + to) / 2;
            }

            // 8. Update castling rights when kings or rooks move or are captured.
            boolean wK = whiteKingSideCastle,  wQ = whiteQueenSideCastle;
            boolean bK = blackKingSideCastle,  bQ = blackQueenSideCastle;
            if (from == 4  || to == 4)  { wK = false; wQ = false; } // White king
            if (from == 60 || to == 60) { bK = false; bQ = false; } // Black king
            if (from == 0  || to == 0)  wQ = false;                 // a1 rook
            if (from == 7  || to == 7)  wK = false;                 // h1 rook
            if (from == 56 || to == 56) bQ = false;                 // a8 rook
            if (from == 63 || to == 63) bK = false;                 // h8 rook

            // 9. Reposition the rook during castling.
            if (movIdx == 5 && from == 4) {           // White king castles
                if (to == 6) {
                    nMg += shiftRook(next, 3, 7, 5, true);
                    nEg += shiftRook(next, 3, 7, 5, false);
                } else if (to == 2) {
                    nMg += shiftRook(next, 3, 0, 3, true);
                    nEg += shiftRook(next, 3, 0, 3, false);
                }
            } else if (movIdx == 11 && from == 60) {  // Black king castles
                if (to == 62) {
                    nMg += shiftRook(next, 9, 63, 61, true);
                    nEg += shiftRook(next, 9, 63, 61, false);
                } else if (to == 58) {
                    nMg += shiftRook(next, 9, 56, 59, true);
                    nEg += shiftRook(next, 9, 56, 59, false);
                }
            }

            Position result = new Position(next, !whiteToMove, wK, wQ, bK, bQ, nextEpSq);
            result.mgScore  = nMg;
            result.egScore  = nEg;
            result.phase    = nPh;
            return result;
        }

        /**
         * Moves rook bitboard {@code idx} from {@code fromSq} to {@code toSq},
         * updates {@code pieces} in place, and returns the net PST score delta.
         * The {@code mg} flag selects midgame (true) or endgame (false) tables.
         */
        private static int shiftRook(long[] pieces, int idx, int fromSq, int toSq, boolean mg) {
            pieces[idx] ^= (1L << fromSq | 1L << toSq);
            return mg ? mgValue(idx, toSq) - mgValue(idx, fromSq)
                      : egValue(idx, toSq) - egValue(idx, fromSq);
        }

        // ── Move generation ───────────────────────────────────────────────────

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

        /** Filters pseudo-legal moves, returning only fully legal ones. */
        MoveList legalMoves() {
            MoveList pseudo = pseudoLegalMoves();
            MoveList legal  = new MoveList();
            for (int i = 0; i < pseudo.count; i++) {
                Position next = makeMove(pseudo.moves[i]);
                if (!next.inCheck(!next.whiteToMove)) legal.add(pseudo.moves[i]);
            }
            return legal;
        }

        void genPawn(MoveList list, boolean w) {
            long enemy = w ? blackPieces : whitePieces;
            long empty = ~allPieces;

            if (w) {
                long single = (wp << 8) & empty;
                addPawnMoves(list, single & ~RANK_8, 8,  false);
                addPawnMoves(list, single &  RANK_8, 8,  true);
                addPawnMoves(list, ((single & RANK_3) << 8) & empty, 16, false);

                long cL = (wp << 7) & enemy & ~FILE_H;
                addPawnMoves(list, cL & ~RANK_8, 7, false);
                addPawnMoves(list, cL &  RANK_8, 7, true);
                long cR = (wp << 9) & enemy & ~FILE_A;
                addPawnMoves(list, cR & ~RANK_8, 9, false);
                addPawnMoves(list, cR &  RANK_8, 9, true);

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

                long cL = (bp >> 9) & enemy & ~FILE_H;
                addPawnMoves(list, cL & ~RANK_1, -9, false);
                addPawnMoves(list, cL &  RANK_1, -9, true);
                long cR = (bp >> 7) & enemy & ~FILE_A;
                addPawnMoves(list, cR & ~RANK_1, -7, false);
                addPawnMoves(list, cR &  RANK_1, -7, true);

                if (enPassantSq != -1) {
                    long ep = 1L << enPassantSq;
                    if (((bp >> 9) & ep & ~FILE_H) != 0) list.add(Move.create(enPassantSq + 9, enPassantSq, 0));
                    if (((bp >> 7) & ep & ~FILE_A) != 0) list.add(Move.create(enPassantSq + 7, enPassantSq, 0));
                }
            }
        }

        void genKnight(MoveList list, boolean w) {
            long pieces   = w ? wn : bn;
            long friendly = w ? whitePieces : blackPieces;
            while (pieces != 0) {
                int from = Long.numberOfTrailingZeros(pieces);
                serializeMoves(list, from, KNIGHT_ATTACKS[from] & ~friendly);
                pieces &= pieces - 1;
            }
        }

        void genKing(MoveList list, boolean w) {
            int  from     = Long.numberOfTrailingZeros(w ? wk : bk);
            long friendly = w ? whitePieces : blackPieces;
            serializeMoves(list, from, KING_ATTACKS[from] & ~friendly);

            if (w) {
                if (whiteKingSideCastle && (allPieces & 0x60L) == 0
                        && !isSquareAttacked(4, false) && !isSquareAttacked(5, false) && !isSquareAttacked(6, false))
                    list.add(Move.create(4, 6, 0));
                if (whiteQueenSideCastle && (allPieces & 0x0EL) == 0
                        && !isSquareAttacked(4, false) && !isSquareAttacked(3, false) && !isSquareAttacked(2, false))
                    list.add(Move.create(4, 2, 0));
            } else {
                if (blackKingSideCastle && (allPieces & 0x6000000000000000L) == 0
                        && !isSquareAttacked(60, true) && !isSquareAttacked(61, true) && !isSquareAttacked(62, true))
                    list.add(Move.create(60, 62, 0));
                if (blackQueenSideCastle && (allPieces & 0x0E00000000000000L) == 0
                        && !isSquareAttacked(60, true) && !isSquareAttacked(59, true) && !isSquareAttacked(58, true))
                    list.add(Move.create(60, 58, 0));
            }
        }

        void genSlidingPieces(MoveList list, boolean w, int[] offsets, long pieces) {
            long friendly = w ? whitePieces : blackPieces;
            long enemy    = w ? blackPieces : whitePieces;
            while (pieces != 0) {
                int from = Long.numberOfTrailingZeros(pieces);
                for (int offset : offsets) {
                    int cur = from;
                    while (true) {
                        int next = cur + offset;
                        if (next < 0 || next >= 64 || isWrap(cur, next, offset)) break;
                        long bit = 1L << next;
                        if ((friendly & bit) != 0) break;
                        list.add(Move.create(from, next, 0));
                        if ((enemy & bit) != 0) break;
                        cur = next;
                    }
                }
                pieces &= pieces - 1;
            }
        }

        private void serializeMoves(MoveList list, int from, long dests) {
            while (dests != 0) {
                list.add(Move.create(from, Long.numberOfTrailingZeros(dests), 0));
                dests &= dests - 1;
            }
        }

        private void addPawnMoves(MoveList list, long dests, int offset, boolean promo) {
            while (dests != 0) {
                int to = Long.numberOfTrailingZeros(dests), from = to - offset;
                if (promo) {
                    list.add(Move.create(from, to, 1)); // Queen
                    list.add(Move.create(from, to, 4)); // Knight
                    list.add(Move.create(from, to, 2)); // Rook
                    list.add(Move.create(from, to, 3)); // Bishop
                } else {
                    list.add(Move.create(from, to, 0));
                }
                dests &= dests - 1;
            }
        }

        // ── Attack detection ──────────────────────────────────────────────────

        /** Returns {@code true} if the specified side's king is in check. */
        boolean inCheck(boolean isWhiteKing) {
            long king = isWhiteKing ? wk : bk;
            if (king == 0) return true; // No king → treat as in check.
            return isSquareAttacked(Long.numberOfTrailingZeros(king), !isWhiteKing);
        }

        /** Returns {@code true} if {@code sq} is attacked by the side specified by {@code byWhite}. */
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

            // Knight and king lookups
            if ((KNIGHT_ATTACKS[sq] & (byWhite ? wn : bn)) != 0) return true;
            if ((KING_ATTACKS[sq]   & (byWhite ? wk : bk)) != 0) return true;

            return isAttackedBySlider(sq, byWhite);
        }

        /**
         * Returns {@code true} if a rook/queen (orthogonal) or bishop/queen
         * (diagonal) owned by {@code byWhite} attacks {@code sq}.
         */
        private boolean isAttackedBySlider(int sq, boolean byWhite) {
            long straight = byWhite ? (wr | wq) : (br | bq);
            long diagonal = byWhite ? (wb | wq) : (bb | bq);

            for (int offset : ROOK_OFFSETS) {
                int cur = sq;
                while (true) {
                    int next = cur + offset;
                    if (next < 0 || next >= 64 || isWrap(cur, next, offset)) break;
                    long bit = 1L << next;
                    if ((straight  & bit) != 0) return true;
                    if ((allPieces & bit) != 0) break; // Ray blocked.
                    cur = next;
                }
            }
            for (int offset : BISHOP_OFFSETS) {
                int cur = sq;
                while (true) {
                    int next = cur + offset;
                    if (next < 0 || next >= 64 || isWrap(cur, next, offset)) break;
                    long bit = 1L << next;
                    if ((diagonal  & bit) != 0) return true;
                    if ((allPieces & bit) != 0) break;
                    cur = next;
                }
            }
            return false;
        }

        /**
         * Returns {@code true} if moving from {@code from} to {@code to} would
         * wrap illegally around the board edge.
         * Vertical offsets (±8) must not change the file; all others move by
         * exactly one file per step.
         */
        private boolean isWrap(int from, int to, int offset) {
            int diff = Math.abs((to & 7) - (from & 7));
            return Math.abs(offset) == 8 ? diff != 0 : diff != 1;
        }

        // ── Board display ─────────────────────────────────────────────────────

        /** Returns the character for the piece on {@code sq}, or {@code '.'}. */
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
                for (int file = 0; file < 8; file++) System.out.print(getPieceAt(rank * 8 + file) + " ");
                System.out.println();
            }
            System.out.println("\n   a b c d e f g h\n");
        }

        // ── FEN parsing helpers ───────────────────────────────────────────────

        private static int getPieceIndex(char c) {
            return switch (c) {
                case 'P' -> 0; case 'N' -> 1; case 'B' -> 2;
                case 'R' -> 3; case 'Q' -> 4; case 'K' -> 5;
                case 'p' -> 6; case 'n' -> 7; case 'b' -> 8;
                case 'r' -> 9; case 'q' -> 10; case 'k' -> 11;
                default  -> -1;
            };
        }

        private static boolean[] parseCastlingRights(String s) {
            return new boolean[]{ s.contains("K"), s.contains("Q"), s.contains("k"), s.contains("q") };
        }

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
