public class Search {

    // -------------------------------------------------------------------------
    // Shared search state
    // -------------------------------------------------------------------------
    static volatile long startTime;
    static volatile long timeLimit;
    static volatile int  lastScore = 0;

    static long moveTimeMs = 10_000;
    static int MAX_DEPTH = 100;

    static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    // -------------------------------------------------------------------------
    // Killer move table
    // -------------------------------------------------------------------------
    private static final int KILLER_MAX_PLY = 128;
    static final int[][] killers = new int[KILLER_MAX_PLY + 1][2];

    static void clearKillers() {
        for (int[] row : killers) {
            row[0] = 0;
            row[1] = 0;
        }
    }

    private static void storeKiller(int ply, int m, Position pos) {
        if (ply < 0 || ply > KILLER_MAX_PLY) return;
        if ((pos.allPieces & (1L << Move.getTo(m))) != 0) return;
        if (Move.getPromo(m) != 0) return;
        if (killers[ply][0] == m) return;
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = m;
    }

    // -------------------------------------------------------------------------
    // Tiny hash move table
    // -------------------------------------------------------------------------
    private static final int HASH_SIZE = Constants.HASH_MOVE_TABLE_SIZE;
    private static final long[] hashKeys = new long[HASH_SIZE];
    private static final int[]  hashMoves = new int[HASH_SIZE];

    private static int hashIndex(long key) {
        return (int) key & (HASH_SIZE - 1);
    }

    static void clearHashMoves() {
        for (int i = 0; i < HASH_SIZE; i++) {
            hashKeys[i] = 0L;
            hashMoves[i] = 0;
        }
    }

    static int probeHashMove(Position pos) {
        int idx = hashIndex(pos.key);
        return hashKeys[idx] == pos.key ? hashMoves[idx] : 0;
    }

    static void storeHashMove(Position pos, int move) {
        if (move == 0) return;
        int idx = hashIndex(pos.key);
        hashKeys[idx] = pos.key;
        hashMoves[idx] = move;
    }

    // -------------------------------------------------------------------------
    // Position history for repetition detection
    // -------------------------------------------------------------------------
    private static final long[] positionHistory = new long[Constants.POSITION_HISTORY_SIZE];
    private static int historySize = 0;

    static void clearHistory() {
        historySize = 0;
    }

    static void recordPosition(Position pos) {
        if (historySize < positionHistory.length) {
            positionHistory[historySize++] = pos.key;
        } else {
            System.arraycopy(positionHistory, 1, positionHistory, 0, positionHistory.length - 1);
            positionHistory[positionHistory.length - 1] = pos.key;
        }
    }

    private static int countOccurrences(long key) {
        int count = 0;
        for (int i = 0; i < historySize; i++) {
            if (positionHistory[i] == key) count++;
        }
        return count;
    }

    static boolean detectThreeRepeats(Position pos, int move) {
        Position next = pos.makeMove(move);
        return countOccurrences(next.key) >= 2;
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
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
    static void orderMoves(Position pos, MoveList list, int prevBest, int hashMove, int nnHint, int ply) {
        int[] scores = new int[list.count];
        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];
            if      (m == prevBest) scores[i] = 20000;
            else if (m == hashMove) scores[i] = 15000;
            else if (m == nnHint)   scores[i] = 10000;
            else                    scores[i] = scoreMove(pos, m, ply);
        }

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

    // -------------------------------------------------------------------------
    // Negamax iterative deepening with root move reuse
    // -------------------------------------------------------------------------
    static int iterativeNegamax(Position pos, int nnHint) {
        clearKillers();
        lastScore = 0;

        MoveList rootPseudo = new MoveList();
        MoveList rootMoves  = new MoveList();
        pos.legalMoves(rootPseudo, rootMoves);
        if (rootMoves.count == 0) return 0;

        int bestMove = rootMoves.moves[0];
        int depth;
        for (depth = 1; depth <= MAX_DEPTH; depth++) {
            int candidate = findBestRoot(pos, rootMoves, depth, bestMove, nnHint);
            if (isTimeUp()) break;
            bestMove = candidate;
            if (Math.abs(lastScore) > 90000) break;
        }
        System.out.println("info string depth: " + depth);
        return bestMove;
    }

    static int findBestRoot(Position pos, MoveList rootMoves, int depth, int prevBest, int nnHint) {
        RootSearchResult result = searchRoot(pos, rootMoves, depth, prevBest, nnHint, true);
        lastScore = result.score;
        return result.bestMove;
    }

    // Untimed helper for the guess thread.
    static int findBest(Position pos, int depth, int prevBest, int nnHint) {
        MoveList pseudo = new MoveList();
        MoveList rootMoves = new MoveList();
        pos.legalMoves(pseudo, rootMoves);
        if (rootMoves.count == 0) return 0;

        int bestMove = rootMoves.moves[0];
        for (int d = 1; d <= depth; d++) {
            RootSearchResult result = searchRoot(pos, rootMoves, d, prevBest, nnHint, false);
            bestMove = result.bestMove;
            prevBest = bestMove;
        }

        return bestMove;
    }

    private static RootSearchResult searchRoot(Position pos, MoveList rootMoves, int depth,
                                               int prevBest, int nnHint, boolean timed) {
        int hashMove = probeHashMove(pos);
        orderMoves(pos, rootMoves, prevBest, hashMove, nnHint, 0);

        int bestMove = rootMoves.moves[0];
        int alpha = -1000000;
        int beta  =  1000000;

        for (int i = 0; i < rootMoves.count; i++) {
            int move = rootMoves.moves[i];
            Position child = pos.makeMove(move);

            int score = searchRootChild(child, depth, alpha, beta, i == 0, timed);

            if (score > alpha) {
                alpha = score;
                bestMove = move;
            }

            if (timed && isTimeUp()) break;
        }

        storeHashMove(pos, bestMove);
        return new RootSearchResult(bestMove, alpha);
    }

    private static int searchRootChild(Position child, int depth, int alpha, int beta,
                                       boolean firstMove, boolean timed) {
        if (firstMove) {
            return -negamax(child, depth - 1, -beta, -alpha, 1, timed);
        }

        int score = -negamax(child, depth - 1, -alpha - 1, -alpha, 1, timed);
        if (score > alpha && score < beta) {
            score = -negamax(child, depth - 1, -beta, -alpha, 1, timed);
        }
        return score;
    }

    private static final class RootSearchResult {
        final int bestMove;
        final int score;

        RootSearchResult(int bestMove, int score) {
            this.bestMove = bestMove;
            this.score = score;
        }
    }

    static int negamax(Position pos, int depth, int alpha, int beta, int ply, boolean timed) {
        if ((timed && isTimeUp()) || depth == 0) return evaluate(pos);

        MoveList moves = pos.pseudoLegalMoves();
        int legals = 0;
        int bestMove = 0;

        int hashMove = probeHashMove(pos);
        orderMoves(pos, moves, 0, hashMove, 0, ply);

        for (int i = 0; i < moves.count; i++) {
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue;
            legals++;

            int score = -negamax(next, depth - 1, -beta, -alpha, ply + 1, timed);

            if (score >= beta) {
                storeKiller(ply, moves.moves[i], pos);
                storeHashMove(pos, moves.moves[i]);
                return beta;
            }
            if (score > alpha) {
                alpha = score;
                bestMove = moves.moves[i];
            }

            if (timed && isTimeUp()) break;
        }

        if (bestMove != 0) storeHashMove(pos, bestMove);
        if (legals == 0) return pos.inCheck(pos.whiteToMove) ? (-100000 + depth) : 0;
        return alpha;
    }

    // -------------------------------------------------------------------------
    // Cheap repetition fallback
    // -------------------------------------------------------------------------
    static int bestNonRepeatingMove(Position pos, int forbiddenMove, int nnHint) {
        MoveList pseudo = new MoveList();
        MoveList moves  = new MoveList();
        pos.legalMoves(pseudo, moves);
        if (moves.count == 0) return 0;

        int hashMove = probeHashMove(pos);
        orderMoves(pos, moves, 0, hashMove, nnHint, 0);

        if (nnHint != 0 && nnHint != forbiddenMove && !detectThreeRepeats(pos, nnHint)) {
            for (int i = 0; i < moves.count; i++) {
                if (moves.moves[i] == nnHint) return nnHint;
            }
        }

        for (int i = 0; i < moves.count; i++) {
            int m = moves.moves[i];
            if (m != forbiddenMove && !detectThreeRepeats(pos, m)) return m;
        }

        for (int i = 0; i < moves.count; i++) {
            int m = moves.moves[i];
            if (m != forbiddenMove) return m;
        }

        return forbiddenMove;
    }

    // -------------------------------------------------------------------------
    // Guess table
    // -------------------------------------------------------------------------
    static class GuessTable {

        private final int[] opponentGuesses = new int[Constants.MAX_GUESSES];
        private final int[] replyMoves      = new int[Constants.MAX_GUESSES];
        private volatile int size           = 0;
        private volatile boolean done       = false;

        int lookupReply(int opponentMove) {
            for (int i = 0; i < size; i++) {
                if (opponentGuesses[i] == opponentMove) return replyMoves[i];
            }
            return 0;
        }

        void finish() { done = true; }
        boolean isDone() { return done; }
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
            MoveList oppPseudo = new MoveList();
            MoveList opponentMoves = new MoveList();
            posAfterOurMove.legalMoves(oppPseudo, opponentMoves);
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

                MoveList ourPseudo = new MoveList();
                MoveList ourMoves  = new MoveList();
                afterOpponent.legalMoves(ourPseudo, ourMoves);
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
                Thread.yield();
            }

            for (int i = 0; i < table.size && !table.isDone(); i++) {
                Position afterOpponent = posAfterOurMove.makeMove(table.opponentGuesses[i]);
                int refined = findBest(afterOpponent, Constants.GUESS_REFINE_DEPTH, table.replyMoves[i], 0);
                if (refined != 0) table.replyMoves[i] = refined;
                Thread.yield();
            }

            table.finish();
        }

        private static void sortByScore(MoveList list, int[] scores) {
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
    }

    static int detectOpponentMove(Position before, Position after) {
        if (before == null || after == null) return 0;
        MoveList pseudo = new MoveList();
        MoveList moves  = new MoveList();
        before.legalMoves(pseudo, moves);
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