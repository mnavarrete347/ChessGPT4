import java.util.Collections;
import java.util.Map;

public class Search {

    static volatile long startTime;
    static volatile long timeLimit;
    static volatile int lastScore = 0;

    static long moveTimeMs = 10_000;
    static int MAX_DEPTH = 100;

    private static final int KILLER_MAX_PLY = 128;
    static final int[][] killers = new int[KILLER_MAX_PLY + 1][2];

    static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    // Killer moves are quiet moves that caused a beta cutoff at the same ply.
    // They are searched early in sibling nodes because they often remain strong.
    private static void storeKiller(int ply, int move, Position pos) {
        if (ply < 0 || ply > KILLER_MAX_PLY) return;
        if ((pos.allPieces & (1L << Move.getTo(move))) != 0) return;
        if (Move.getPromo(move) != 0) return;
        if (killers[ply][0] == move) return;
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = move;
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
// Move ordering priority (highest → lowest)
//
// 20000  Previous best move (PV move from last iteration)
// 10000  NN top policy moves (ranked)
//  9500  NN hint (from guess thread)
//  9000  NN second-best move
//  8000  NN third-best move
//  7000  NN fourth-best move
// 1000+  Captures (MVV-LVA)
//        → Most Valuable Victim - Least Valuable Attacker
//        → winning captures searched early
//  901   Killer move #1 (per ply)
//  900   Killer move #2 (per ply)
//        → quiet moves that caused beta cutoffs earlier
// +900   Bonus added for promotions (applies on top of any category)
//    0   All other quiet moves
//
// Notes:
// - Ordering is critical for alpha-beta pruning efficiency
// - Earlier moves are searched first, increasing chances of cutoffs
// - NN moves are only applied at root search
// -------------------------------------------------------------------------
    static void orderMoves(Position pos, MoveList list, int prevBest, int nnHint, int ply) {
        orderMoves(pos, list, prevBest, nnHint, ply, new int[0]);
    }

    // Root ordering can optionally use a small set of neural-policy scores.
    // Everywhere else we fall back to PV move, captures, promotions, and killers.
    static void orderMoves(Position pos, MoveList list, int prevBest, int nnHint, int ply,
                           int[] nnMoves) {
        int[] scores = new int[list.count];

        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];

            if (m == prevBest)    scores[i] = 20000;
            else if (m == nnHint) scores[i] = 9500;
            else {
                int nnScore = -1;

                // Small loop to score each nn moves
                for (int k = 0; k < nnMoves.length; k++) {
                    if (nnMoves[k] == m) {
                        nnScore = 10000 - (k * 1000);
                        break;
                    }
                }
                if (nnScore != -1) {
                    scores[i] = nnScore;
                } else {
                    scores[i] = scoreMove(pos, m, ply);
                }
            }
        }

        // Simple selection sort
        for (int i = 0; i < list.count - 1; i++) {
            for (int j = i + 1; j < list.count; j++) {
                if (scores[j] > scores[i]) {
                    int t = list.moves[i];
                    list.moves[i] = list.moves[j];
                    list.moves[j] = t;

                    int s = scores[i];
                    scores[i] = scores[j];
                    scores[j] = s;
                }
            }
        }
    }

    private static int scoreMove(Position pos, int move, int ply) {
        int score = 0;
        int to = Move.getTo(move);
        int from = Move.getFrom(move);

        if ((pos.allPieces & (1L << to)) != 0) {
            score = 1000 + pieceValue(pos.getPieceAt(to)) - (pieceValue(pos.getPieceAt(from)) / 10);
        } else if (ply >= 0 && ply <= KILLER_MAX_PLY) {
            if      (killers[ply][0] == move) score = 901;
            else if (killers[ply][1] == move) score = 900;
        }

        if (Move.getPromo(move) != 0) score += 900;
        return score;
    }

    private static int pieceValue(char p) {
        return switch (Character.toUpperCase(p)) {
            case 'P' -> Constants.PAWN;
            case 'N' -> Constants.KNIGHT;
            case 'B' -> Constants.BISHOP;
            case 'R' -> Constants.ROOK;
            case 'Q' -> Constants.QUEEN;
            case 'K' -> 10_000;
            default  -> 0;
        };
    }

    static int iterativeNegamax(Position pos, int nnHint, NeuralEngine nn) {
        for (int[] row : killers) {
            row[0] = 0;
            row[1] = 0;
        }
        lastScore = 0;

        int bestMove = 0;
        int depth;
        for (depth = 1; depth <= MAX_DEPTH; depth++) {
            int candidate = findBest(pos, depth, bestMove, nnHint, nn, true);
            if (isTimeUp()) break;

            bestMove = candidate;

            if (Math.abs(lastScore) > 90000) break;
        }
        //printSearchInfo(depth, bestMove, lastScore);
        return bestMove;
    }

    private static void printSearchInfo(int depth, int bestMove, int score) {
        long elapsedMs = Math.max(1L, System.currentTimeMillis() - startTime);
        String scoreType = Math.abs(score) > 90000 ? "mate" : "cp";
        int scoreValue = Math.abs(score) > 90000 ? (100000 - Math.abs(score)) : score;
        System.out.println("info depth " + depth
                + " score " + scoreType + " " + scoreValue
                + " time " + elapsedMs
                + " pv " + Move.toUci(bestMove));
    }

    static int findBest(Position pos, int depth, int prevBest, int nnHint, NeuralEngine nn, boolean timed) {
        MoveList moves = pos.legalMoves();
        if (moves.count == 0) return 0;

        int[] nnMoves = new int[0];
        if (nn != null && timed) {
            try {
                nnMoves = nn.topPolicyMoves(pos, moves, Constants.NN_TOP_K);
            } catch (Exception ignored) {}
        }

        orderMoves(pos, moves, prevBest, nnHint, 0, nnMoves);

        int bestMove = moves.moves[0];
        int alpha = -1000000;
        int beta = 1000000;

        for (int i = 0; i < moves.count; i++) {
            int move = moves.moves[i];
            Position child = pos.makeMove(move);
            int score = -negamax(child, depth - 1, -beta, -alpha, 1, timed);

            if (score > alpha) {
                alpha = score;
                bestMove = move;
                lastScore = score;
            }

            if (timed && isTimeUp()) break;
        }
        return bestMove;
    }

    static int negamax(Position pos, int depth, int alpha, int beta, int ply, boolean timed) {
        if ((timed && isTimeUp()) || depth == 0) return evaluate(pos);

        MoveList moves = pos.pseudoLegalMoves();
        int legals = 0;
        orderMoves(pos, moves, 0, 0, ply);

        for (int i = 0; i < moves.count; i++) {
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue;
            legals++;

            int score = -negamax(next, depth - 1, -beta, -alpha, ply + 1, timed);
            if (score >= beta) {
                storeKiller(ply, moves.moves[i], pos);
                return beta;
            }
            if (score > alpha) alpha = score;

            if (timed && isTimeUp()) break;
        }

        if (legals == 0) return pos.inCheck(pos.whiteToMove) ? (-100000 + depth) : 0;
        return alpha;
    }

    static int findBestNonRepeatingMove(Position pos, int forbiddenMove, int nnHint, NeuralEngine nn) {
        MoveList moves = pos.legalMoves();
        if (moves.count == 0) return 0;

        int[] nnMoves = new int[0];

        if (nn != null) {
            try {
                nnMoves = nn.topPolicyMoves(pos, moves, Constants.NN_TOP_K);
            } catch (Exception ignored) {
            }
        }

        orderMoves(pos, moves, 0, nnHint, 0, nnMoves);

        if (nnHint != 0 && nnHint != forbiddenMove) {
            for (int i = 0; i < moves.count; i++) {
                if (moves.moves[i] == nnHint) return nnHint;
            }
        }

        for (int i = 0; i < moves.count; i++) {
            int m = moves.moves[i];
            if (m != forbiddenMove) return m;
        }

        return forbiddenMove;
    }

    static class GuessTable {
        private final int[] opponentGuesses = new int[Constants.MAX_GUESSES];
        private final int[] replyMoves = new int[Constants.MAX_GUESSES];
        private volatile int size = 0;
        private volatile boolean done = false;

        int lookupReply(int opponentMove) {
            for (int i = 0; i < size; i++) {
                if (opponentGuesses[i] == opponentMove) return replyMoves[i];
            }
            return 0;
        }

        void finish() {
            done = true;
        }

        boolean isDone() {
            return done;
        }
    }

    static class GuessingThread extends Thread {

        private final Position posAfterOurMove;
        private final NeuralEngine nn;
        private final GuessTable table;

        GuessingThread(Position posAfterOurMove, NeuralEngine nn, GuessTable table) {
            this.posAfterOurMove = posAfterOurMove;
            this.nn = nn;
            this.table = table;
            setDaemon(true);
            setPriority(Thread.MIN_PRIORITY);
        }

        @Override
        public void run() {
            MoveList opponentMoves = posAfterOurMove.legalMoves();
            if (opponentMoves.count == 0) {
                table.finish();
                return;
            }

            int[] scores = new int[opponentMoves.count];
            for (int i = 0; i < opponentMoves.count; i++) {
                scores[i] = -evaluate(posAfterOurMove.makeMove(opponentMoves.moves[i]));
            }
            sortByScore(opponentMoves, scores);

            int generated = 0;
            for (int i = 0; i < opponentMoves.count && !table.isDone() && generated < Constants.MAX_GUESSES; i++) {
                int opponentMove = opponentMoves.moves[i];
                Position afterOpponent = posAfterOurMove.makeMove(opponentMove);
                MoveList ourMoves = afterOpponent.legalMoves();
                if (ourMoves.count == 0) continue;

                int reply = 0;
                try {
                    reply = nn.topPolicyMove(afterOpponent, ourMoves);
                } catch (Exception e) {
                    continue;
                }

                if (reply != 0) {
                    table.opponentGuesses[generated] = opponentMove;
                    table.replyMoves[generated] = reply;
                    table.size = generated + 1;
                    generated++;
                }
                Thread.yield();
            }

            for (int i = 0; i < table.size && !table.isDone(); i++) {
                Position afterOpponent = posAfterOurMove.makeMove(table.opponentGuesses[i]);
                int seed = table.replyMoves[i];
                int refined = findBest(afterOpponent, Constants.GUESS_REFINE_DEPTH, seed, 0, nn, false);
                if (refined != 0) table.replyMoves[i] = refined;
                Thread.yield();
            }

//            if (generated > 0) {
//                System.out.println("info string guess count=" + generated);
//            }
            table.finish();
        }

        private static void sortByScore(MoveList list, int[] scores) {
            for (int i = 0; i < list.count - 1; i++) {
                for (int j = i + 1; j < list.count; j++) {
                    if (scores[j] > scores[i]) {
                        int tm = list.moves[i];
                        list.moves[i] = list.moves[j];
                        list.moves[j] = tm;

                        int ts = scores[i];
                        scores[i] = scores[j];
                        scores[j] = ts;
                    }
                }
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
