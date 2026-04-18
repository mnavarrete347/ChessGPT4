public class Search {

    // -------------------------------------------------------------------------
    // Shared search state
    // -------------------------------------------------------------------------
    static volatile long    startTime;
    static volatile long    timeLimit;
    static volatile int     lastScore     = 0;

    static int  maxDepth   = 100;
    static long moveTimeMs = 10_000;

    static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    // -------------------------------------------------------------------------
    // PST evaluation
    // -------------------------------------------------------------------------

    static int evaluate(Position pos) {
        int phase = Math.clamp(pos.phase, 0, Constants.MAX_PHASE);
        int score = (pos.mgScore * phase + pos.egScore * (Constants.MAX_PHASE - phase)) / Constants.MAX_PHASE;
        return pos.whiteToMove ? score : -score;
    }

    // -------------------------------------------------------------------------
    // Move ordering
    // -------------------------------------------------------------------------
    static void orderMoves(Position pos, MoveList list, int prevBest, int nnHint) {
        int[] scores = new int[list.count];
        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];
            if (m == prevBest) scores[i] = 20000;
            else if (m == nnHint) scores[i] = 10000;
            else scores[i] = scoreMove(pos, m);
        }
        // Simple selection sort for move ordering
        for (int i = 0; i < list.count - 1; i++) {
            for (int j = i + 1; j < list.count; j++) {
                if (scores[j] > scores[i]) {
                    int t = list.moves[i]; list.moves[i] = list.moves[j]; list.moves[j] = t;
                    int s = scores[i]; scores[i] = scores[j]; scores[j] = s;
                }
            }
        }
    }

    private static int scoreMove(Position pos, int move) {
        int score = 0, to = Move.getTo(move), from = Move.getFrom(move);
        if ((pos.allPieces & (1L << to)) != 0)
            score = 1000 + pieceValue(pos.getPieceAt(to)) - (pieceValue(pos.getPieceAt(from)) / 10);
        if (Move.getPromo(move) != 0) score += 900;
        return score;
    }

    private static int pieceValue(char p) {
        return switch (Character.toUpperCase(p)) {
            case 'P' -> Constants.MG_PAWN;   case 'N' -> Constants.MG_KNIGHT;
            case 'B' -> Constants.MG_BISHOP; case 'R' -> Constants.MG_ROOK;
            case 'Q' -> Constants.MG_QUEEN;  case 'K' -> 10_000;
            default  -> 0;
        };
    }

    // -------------------------------------------------------------------------
    // Negamax iterative deepening
    // -------------------------------------------------------------------------
    // nnHint != 0 promotes that move to the front of root move ordering.
    static int iterativeNegamax(Position pos, int nnHint) {
        int bestMove = 0;
        for (int depth = 1; depth <= maxDepth; depth++) {
            int candidate = findBest(pos, depth, bestMove, nnHint);
            if (isTimeUp()) break;
            bestMove = candidate;
            if (Math.abs(lastScore) > 90000) break;
        }
        //System.out.println("info string depth: " + depth);
        // Always return the best move found, never nnHint directly
        return bestMove;
    }

    static int findBest(Position pos, int depth, int prevBest, int nnHint) {
        MoveList moves = pos.legalMoves();
        if (moves.count == 0) return 0;

        orderMoves(pos, moves, prevBest, nnHint);

        int bestMove = moves.moves[0], alpha = -1000000, beta = 1000000;
        for (int i = 0; i < moves.count; i++) {
            int score = -negamax(pos.makeMove(moves.moves[i]), depth - 1, -beta, -alpha);
            if (score > alpha) {
                alpha = score;
                bestMove = moves.moves[i];
                lastScore = score;
            }
            if (isTimeUp()) break;
        }
        return bestMove;
    }

    static int negamax(Position pos, int depth, int alpha, int beta) {
        if (isTimeUp() || depth == 0) return evaluate(pos);

        MoveList moves = pos.pseudoLegalMoves();
        int legals = 0;
        orderMoves(pos, moves, 0, 0);

        for (int i = 0; i < moves.count; i++) {
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue;
            legals++;
            int score = -negamax(next, depth - 1, -beta, -alpha);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        if (legals == 0) return pos.inCheck(pos.whiteToMove) ? (-100000 + depth) : 0;
        return alpha;
    }

    // -------------------------------------------------------------------------
    // Guess table — built during the opponent's turn using the NN policy head.
    // Stores up to MAX_GUESSES (opponentMove, ourReply) pairs.
    // -------------------------------------------------------------------------

    static final int MAX_GUESSES = 10;

    static class GuessTable {

        private final int[]   opponentGuesses = new int[MAX_GUESSES];
        private final int[]   replyMoves      = new int[MAX_GUESSES];
        private volatile int  size            = 0;
        private volatile boolean done         = false;

        // Returns the pre-computed reply for opponentMove, or 0 if not found.
        int lookupReply(int opponentMove) {
            for (int i = 0; i < size; i++)
                if (opponentGuesses[i] == opponentMove) return replyMoves[i];
            return 0;
        }

        // Called by the background thread to stop generating new pairs.
        void finish() { done = true; }

        boolean isDone() { return done; }
    }

    // Background thread that populates a GuessTable while the opponent thinks.
    // For each guessed opponent move the NN policy head provides a reply suggestion.
    // Pairs are generated until MAX_GUESSES is reached or finish() is called.
    static class GuessingThread extends Thread {

        private final Position    posAfterOurMove; // position after we played our move
        private final NeuralEngine nn;
        private final GuessTable   table;

        GuessingThread(Position posAfterOurMove, NeuralEngine nn, GuessTable table) {
            this.posAfterOurMove = posAfterOurMove;
            this.nn              = nn;
            this.table           = table;
            setDaemon(true);
            setPriority(Thread.MIN_PRIORITY);
        }

        @Override
        public void run() {
            MoveList opponentMoves = posAfterOurMove.legalMoves();
            if (opponentMoves.count == 0) { table.finish(); return; }

            // Order opponent moves by their PST value so we guess plausible moves first.
            // We evaluate from the opponent's point of view (negate evaluate).
            int[] scores = new int[opponentMoves.count];
            for (int i = 0; i < opponentMoves.count; i++)
                scores[i] = -evaluate(posAfterOurMove.makeMove(opponentMoves.moves[i]));
            sortByScore(opponentMoves, scores);

            int generated = 0;
            for (int i = 0; i < opponentMoves.count && !table.isDone() && generated < MAX_GUESSES; i++) {
                int opponentMove = opponentMoves.moves[i];
                Position afterOpponent = posAfterOurMove.makeMove(opponentMove);

                // Ask the NN for our best reply to this opponent move.
                MoveList ourMoves = afterOpponent.legalMoves();
                if (ourMoves.count == 0) continue;

                int reply;
                try {
                    reply = nn.topPolicyMove(afterOpponent, ourMoves);
                } catch (Exception e) {
                    // NN unavailable for this position — skip this pair.
                    continue;
                }

                if (reply != 0) {
                    table.opponentGuesses[generated] = opponentMove;
                    table.replyMoves[generated]      = reply;
                    table.size                       = generated + 1;
                    generated++;
//                    System.out.println("info string guess[" + generated + "] opp="
//                            + Move.toUci(opponentMove) + " reply=" + Move.toUci(reply));
                }
            }

            table.finish();
        }

        private static void sortByScore(MoveList list, int[] scores) {
            for (int i = 0; i < list.count - 1; i++)
                for (int j = i + 1; j < list.count; j++)
                    if (scores[j] > scores[i]) {
                        int t = list.moves[i]; list.moves[i] = list.moves[j]; list.moves[j] = t;
                        int s = scores[i];     scores[i]     = scores[j];     scores[j]     = s;
                    }
        }
    }

    // Detects which legal move from 'before' produced 'after' by comparing board state.
    static int detectOpponentMove(Position before, Position after) {
        if (before == null || after == null) return 0;
        MoveList moves = before.legalMoves();
        for (int i = 0; i < moves.count; i++) {
            Position result = before.makeMove(moves.moves[i]);
            if (result.allPieces  == after.allPieces
                    && result.wp  == after.wp  && result.bp == after.bp
                    && result.wn  == after.wn  && result.bn == after.bn
                    && result.wr  == after.wr  && result.br == after.br
                    && result.whiteToMove == after.whiteToMove)
                return moves.moves[i];
        }
        return 0;
    }
}
