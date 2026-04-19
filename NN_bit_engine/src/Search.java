public class Search {

    // -------------------------------------------------------------------------
    // Shared search state
    // -------------------------------------------------------------------------
    static volatile long startTime;
    static volatile long timeLimit;
    static volatile int  lastScore = 0;

    static int  maxDepth   = 100;
    static long moveTimeMs = 10_000;

    static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    static int evaluate(Position pos) {
        int eval = pos.score;

        if (pos.totalMaterial() < 2200) {
            int whiteKingSq = Long.numberOfTrailingZeros(pos.wk);
            int blackKingSq = Long.numberOfTrailingZeros(pos.bk);
            eval += kingCentralBonus(whiteKingSq);
            eval -= kingCentralBonus(blackKingSq);
        }
        return pos.whiteToMove ? eval : -eval;
    }

    static int kingCentralBonus(int sq) {
        int file = sq % 8;
        int rank = sq / 8;
        int distFromCenter = Math.abs(file - 3) + Math.abs(rank - 3);
        return (7 - distFromCenter) * 5;
    }

    // -------------------------------------------------------------------------
    // Move ordering
    // -------------------------------------------------------------------------
    static void orderMoves(Position pos, MoveList list, int prevBest, int nnHint) {
        int[] scores = new int[list.count];
        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];
            if (m == prevBest)     scores[i] = 20000;
            else if (m == nnHint)  scores[i] = 10000;
            else                   scores[i] = scoreMove(pos, m);
        }
        for (int i = 0; i < list.count - 1; i++)
            for (int j = i + 1; j < list.count; j++)
                if (scores[j] > scores[i]) {
                    int t = list.moves[i]; list.moves[i] = list.moves[j]; list.moves[j] = t;
                    int s = scores[i];     scores[i]     = scores[j];     scores[j]     = s;
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
            case 'P' -> Constants.PAWN;   case 'N' -> Constants.KNIGHT;
            case 'B' -> Constants.BISHOP; case 'R' -> Constants.ROOK;
            case 'Q' -> Constants.QUEEN;  case 'K' -> 10_000;
            default  -> 0;
        };
    }

    // -------------------------------------------------------------------------
    // Negamax iterative deepening
    // -------------------------------------------------------------------------
    static int iterativeNegamax(Position pos, int nnHint) {
        int bestMove = 0;
        for (int depth = 1; depth <= maxDepth; depth++) {
            int candidate = findBest(pos, depth, bestMove, nnHint);
            if (isTimeUp()) break;
            bestMove = candidate;
            if (Math.abs(lastScore) > 90000) break;
        }
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
                alpha    = score;
                bestMove = moves.moves[i];
                lastScore = score;
            }
            if (isTimeUp()) break;
        }
        return bestMove;
    }

    static int negamax(Position pos, int depth, int alpha, int beta) {
        if (isTimeUp() || depth == 0) return evaluate(pos);

        MoveList moves  = pos.pseudoLegalMoves();
        int      legals = 0;
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
    // Guess table
    // -------------------------------------------------------------------------

    static class GuessTable {

        private final int[]          opponentGuesses = new int[Constants.MAX_GUESSES];
        private final int[]          replyMoves      = new int[Constants.MAX_GUESSES];
        private volatile int         size            = 0;
        private volatile boolean     done            = false;

        int lookupReply(int opponentMove) {
            for (int i = 0; i < size; i++)
                if (opponentGuesses[i] == opponentMove) return replyMoves[i];
            return 0;
        }

        void    finish()  { done = true; }
        boolean isDone()  { return done; }
    }

    static class GuessingThread extends Thread {

        private final Position     posAfterOurMove;
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

            int[] scores = new int[opponentMoves.count];
            for (int i = 0; i < opponentMoves.count; i++)
                scores[i] = -evaluate(posAfterOurMove.makeMove(opponentMoves.moves[i]));
            sortByScore(opponentMoves, scores);

            int generated = 0;
            for (int i = 0; i < opponentMoves.count && !table.isDone() && generated < Constants.MAX_GUESSES; i++) {
                int      opponentMove  = opponentMoves.moves[i];
                Position afterOpponent = posAfterOurMove.makeMove(opponentMove);

                MoveList ourMoves = afterOpponent.legalMoves();
                if (ourMoves.count == 0) continue;

                int reply;
                try {
                    reply = nn.topPolicyMove(afterOpponent, ourMoves);
                } catch (Exception e) {
                    continue;
                }

                if (reply != 0) {
                    table.opponentGuesses[generated] = opponentMove;
                    table.replyMoves[generated]      = reply;
                    table.size                       = generated + 1;

                    System.out.println("info string guess[" + generated + "] opp="
                            + Move.toUci(opponentMove) + " reply=" + Move.toUci(reply));

                    generated++;
                }

                // Yield after each inference so other threads (and the OS) get CPU time between calls.
                Thread.yield();
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

    static int detectOpponentMove(Position before, Position after) {
        if (before == null || after == null) return 0;
        MoveList moves = before.legalMoves();
        for (int i = 0; i < moves.count; i++) {
            Position result = before.makeMove(moves.moves[i]);
            if (samePosition(result, after)) return moves.moves[i];
        }
        return 0;
    }

    static boolean samePosition(Position a, Position b) {
        return a.wp == b.wp && a.wn == b.wn && a.wb == b.wb &&
                a.wr == b.wr && a.wq == b.wq && a.wk == b.wk &&
                a.bp == b.bp && a.bn == b.bn && a.bb == b.bb &&
                a.br == b.br && a.bq == b.bq && a.bk == b.bk &&
                a.whiteToMove == b.whiteToMove &&
                a.whiteKingSideCastle == b.whiteKingSideCastle &&
                a.whiteQueenSideCastle == b.whiteQueenSideCastle &&
                a.blackKingSideCastle == b.blackKingSideCastle &&
                a.blackQueenSideCastle == b.blackQueenSideCastle;
    }
}
