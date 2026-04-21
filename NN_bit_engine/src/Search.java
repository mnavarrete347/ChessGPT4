public class Search {

    // =========================================================================
    // Shared search state — written by Main before each search, read everywhere
    // =========================================================================

    static volatile long startTime;
    static volatile long timeLimit;
    static volatile int  lastScore = 0;

    static long moveTimeMs = 10_000;
    static int  MAX_DEPTH  = 100;

    static boolean isTimeUp() {
        return System.currentTimeMillis() - startTime > timeLimit;
    }

    // =========================================================================
    // Killer move table
    // Two quiet moves per ply that caused a beta cutoff in a sibling branch.
    // Slot 0 is newer; slot 1 is the previous killer, bumped down on each store.
    // =========================================================================

    private static final int KILLER_MAX_PLY = 128;
    static final int[][] killers = new int[KILLER_MAX_PLY + 1][2];

    // Records a quiet beta-cutoff move as a killer at this ply.
    private static void storeKiller(int ply, int m, Position pos) {
        if (ply < 0 || ply > KILLER_MAX_PLY) return;
        if ((pos.allPieces & (1L << Move.getTo(m))) != 0) return; // skip captures
        if (Move.getPromo(m) != 0) return;                        // skip promotions
        if (killers[ply][0] == m) return;                         // already primary, skip
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = m;
    }

    // =========================================================================
    // Hash move table
    // Used as an extra move ordering hint on top of killers.
    // =========================================================================

    private static final int HASH_SIZE = Constants.HASH_MOVE_TABLE_SIZE;
    private static final long[] hashKeys  = new long[HASH_SIZE];
    private static final int[]  hashMoves = new int[HASH_SIZE];

    // Power-of-two size allows cheap bitwise index instead of modulo.
    private static int hashIndex(long key) { return (int) key & (HASH_SIZE - 1); }

    static void clearHashMoves() {
        for (int i = 0; i < HASH_SIZE; i++) { hashKeys[i] = 0L; hashMoves[i] = 0; }
    }

    static int probeHashMove(Position pos) {
        int idx = hashIndex(pos.key);
        return hashKeys[idx] == pos.key ? hashMoves[idx] : 0;
    }

    static void storeHashMove(Position pos, int move) {
        if (move == 0) return;
        int idx = hashIndex(pos.key);
        hashKeys[idx]  = pos.key;
        hashMoves[idx] = move;
    }

    // =========================================================================
    // Position history for three-fold repetition detection
    // =========================================================================

    private static final long[] positionHistory = new long[Constants.POSITION_HISTORY_SIZE];
    public  static int historySize = 0;

    static void recordPosition(Position pos) {
        if (historySize < positionHistory.length) {
            positionHistory[historySize++] = pos.key;
        } else {
            // Shift left to drop the oldest entry, then append.
            System.arraycopy(positionHistory, 1, positionHistory, 0, positionHistory.length - 1);
            positionHistory[positionHistory.length - 1] = pos.key;
        }
    }

    // Returns true if making this move would produce a position seen twice already.
    static boolean detectThreeRepeats(Position pos, int move) {
        long nextKey = pos.makeMove(move).key;
        int count = 0;
        for (int i = 0; i < historySize; i++) {
            if (positionHistory[i] == nextKey) count++;
        }
        return count >= 2;
    }

    // =========================================================================
    // Evaluation
    // Returns score in centipawns relative to the side to move.
    // Uses the incrementally maintained pos.score, adding a king centrality
    // bonus in endgame positions (low total material).
    // =========================================================================

    static int evaluate(Position pos) {
        int eval = pos.score;

        if (pos.totalMaterial() < 2200) {
            // Reward kings for moving toward the center in the endgame.
            // distance from center d4/d5/e4/e5 (files 3-4, ranks 3-4).
            int wFile = Long.numberOfTrailingZeros(pos.wk) % 8;
            int wRank = Long.numberOfTrailingZeros(pos.wk) / 8;
            int bFile = Long.numberOfTrailingZeros(pos.bk) % 8;
            int bRank = Long.numberOfTrailingZeros(pos.bk) / 8;
            eval += (7 - (Math.abs(wFile - 3) + Math.abs(wRank - 3))) * 5;
            eval -= (7 - (Math.abs(bFile - 3) + Math.abs(bRank - 3))) * 5;
        }

        return pos.whiteToMove ? eval : -eval;
    }

    // =========================================================================
    // Incremental move scoring
    //
    // Priority bands (higher = searched first):
    //   20000  PV / previous iteration best move
    //   15000  Hash move from the transposition table
    //   10000  NN hint from the guess table
    //    1000+ Captures, scored by MVV-LVA (most valuable victim, least valuable attacker)
    //     901  Killer move slot 0
    //     900  Killer move slot 1
    //       0  All other quiet moves
    // =========================================================================

    static int[] scoreMoves(Position pos, MoveList list, int prevBest, int hashMove, int nnHint, int ply) {
        int[] scores = new int[list.count];

        for (int i = 0; i < list.count; i++) {
            int m = list.moves[i];
            if      (m == prevBest) scores[i] = 20000;
            else if (m == hashMove) scores[i] = 15000;
            else if (m == nnHint)   scores[i] = 10000;
            else {
                int to   = Move.getTo(m);
                int from = Move.getFrom(m);
                if ((pos.allPieces & (1L << to)) != 0) {
                    scores[i] = 1000 + pieceValue(pos.getPieceAt(to)) - (pieceValue(pos.getPieceAt(from)) / 10);
                } else if (ply >= 0 && ply <= KILLER_MAX_PLY) {
                    if      (killers[ply][0] == m) scores[i] = 901;
                    else if (killers[ply][1] == m) scores[i] = 900;
                }
                if (Move.getPromo(m) != 0) scores[i] += 900;
            }
        }
        return scores;
    }

    private static void pickNextMove(MoveList moves, int[] scores, int start) {
        int best = start;
        for (int i = start + 1; i < moves.count; i++) {
            if (scores[i] > scores[best]) best = i;
        }

        if (best != start) {
            int tm = moves.moves[start];
            moves.moves[start] = moves.moves[best];
            moves.moves[best] = tm;

            int ts = scores[start];
            scores[start] = scores[best];
            scores[best] = ts;
        }
    }

    private static int pieceValue(char p) {
        return switch (Character.toUpperCase(p)) {
            case 'P' -> Constants.PAWN;   case 'N' -> Constants.KNIGHT;
            case 'B' -> Constants.BISHOP; case 'R' -> Constants.ROOK;
            case 'Q' -> Constants.QUEEN;  case 'K' -> 10_000;
            default  -> 0;
        };
    }

    // =========================================================================
    // Iterative deepening negamax — main search entry point
    // =========================================================================

    // Reusable record to return both the best move and its score from searchRoot.
    private record SearchResult(int bestMove, int score) {}

    static int iterativeNegamax(Position pos, int nnHint) {
        // Clear state from the previous search so stale data does not influence ordering.
        for (int[] row : killers) { row[0] = 0; row[1] = 0; }
        lastScore = 0;

        MoveList rootPseudo = new MoveList();
        MoveList rootMoves  = new MoveList();
        pos.legalMoves(rootPseudo, rootMoves);
        if (rootMoves.count == 0) return 0;

        // Seed bestMove with any legal move so we always return something valid.
        int bestMove = rootMoves.moves[0];

        for (int depth = 1; depth <= MAX_DEPTH; depth++) {
            SearchResult result = searchRoot(pos, rootMoves, depth, bestMove, nnHint, true);
            lastScore = result.score();

            // Discard a result from an interrupted depth
            if (isTimeUp()) break;

            bestMove = result.bestMove();
            if (Math.abs(lastScore) > 90000) break; // forced mate found
        }

        return bestMove;
    }

    // Untimed iterative deepening used by the guess thread to refine reply moves.
    // Runs all depths up to `depth` without a time check.
    static int findBest(Position pos, int depth, int prevBest, int nnHint) {
        MoveList pseudo    = new MoveList();
        MoveList rootMoves = new MoveList();
        pos.legalMoves(pseudo, rootMoves);
        if (rootMoves.count == 0) return 0;

        int bestMove = rootMoves.moves[0];
        for (int d = 1; d <= depth; d++) {
            SearchResult result = searchRoot(pos, rootMoves, d, prevBest, nnHint, false);
            bestMove = result.bestMove();
            prevBest = bestMove; // carry PV forward into the next depth
        }

        return bestMove;
    }

    // Root search for one depth. Applies principal-variation search (PVS):
    // the first move gets a full window; all others get a null window first,
    // and only re-searched with the full window if they beat alpha.
    private static SearchResult searchRoot(Position pos, MoveList rootMoves, int depth,
                                           int prevBest, int nnHint, boolean timed) {
        int hashMove = probeHashMove(pos);
        int[] scores = scoreMoves(pos, rootMoves, prevBest, hashMove, nnHint, 0);

        int bestMove = rootMoves.moves[0];
        int alpha    = -1_000_000;
        int beta     =  1_000_000;

        for (int i = 0; i < rootMoves.count; i++) {
            pickNextMove(rootMoves, scores, i);
            int move  = rootMoves.moves[i];
            Position child = pos.makeMove(move);

            int score;
            if (i == 0) {
                // First move: full window search.
                score = -negamax(child, depth - 1, -beta, -alpha, 1, timed);
            } else {
                // Subsequent moves: null-window probe, then full re-search if it beats alpha.
                score = -negamax(child, depth - 1, -alpha - 1, -alpha, 1, timed);
                if (score > alpha && score < beta)
                    score = -negamax(child, depth - 1, -beta, -alpha, 1, timed);
            }
            if (score > alpha) {
                alpha    = score;
                bestMove = move;
            }
            if (timed && isTimeUp()) break;
        }
        storeHashMove(pos, bestMove);
        return new SearchResult(bestMove, alpha);
    }

    // Recursive negamax with alpha-beta pruning.
    // `ply` tracks distance from the root for killer move indexing.
    // `timed` is false when called from the guess thread so it never reads the clock.
    static int negamax(Position pos, int depth, int alpha, int beta, int ply, boolean timed) {
        if ((timed && isTimeUp()) || depth == 0) return evaluate(pos);

        MoveList moves  = pos.pseudoLegalMoves(new MoveList());
        int legals   = 0;
        int bestMove = 0;

        int hashMove = probeHashMove(pos);
        int[] scores = scoreMoves(pos, moves, 0, hashMove, 0, ply);

        for (int i = 0; i < moves.count; i++) {
            pickNextMove(moves, scores, i);
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue; // filter pseudo-legal moves
            legals++;

            int score = -negamax(next, depth - 1, -beta, -alpha, ply + 1, timed);

            if (score >= beta) {
                // Beta cutoff: this move is too good for the opponent to allow.
                // Record it as a killer for future sibling nodes at the same ply.
                storeKiller(ply, moves.moves[i], pos);
                if (timed) storeHashMove(pos, moves.moves[i]);
                return beta;
            }
            if (score > alpha) {
                alpha    = score;
                bestMove = moves.moves[i];
            }
            if (timed && isTimeUp()) break;
        }
        if (timed && bestMove != 0) storeHashMove(pos, bestMove);
        if (legals == 0) return pos.inCheck(pos.whiteToMove) ? (-100_000 + depth) : 0;
        return alpha;
    }

    // =========================================================================
    // Repetition avoidance fallback
    // Called by Main when a repeating move pattern is detected.
    // Tries moves in order: NN hint first, then any non-repeating move,
    // then any move that differs from the forbidden move as a last resort.
    // =========================================================================

    static int bestNonRepeatingMove(Position pos, int forbiddenMove, int nnHint) {
        MoveList pseudo = new MoveList();
        MoveList moves  = new MoveList();
        pos.legalMoves(pseudo, moves);
        if (moves.count == 0) return 0;

        int hashMove = probeHashMove(pos);
        int[] scores = scoreMoves(pos, moves, 0, hashMove, nnHint, 0);

        // Try the NN hint first if it avoids both the forbidden move and repetition.
        if (nnHint != 0 && nnHint != forbiddenMove && !detectThreeRepeats(pos, nnHint)) {
            for (int i = 0; i < moves.count; i++) {
                if (moves.moves[i] == nnHint) return nnHint;
            }
        }

        // Walk the moves in score order without fully sorting them up front.
        for (int i = 0; i < moves.count; i++) {
            pickNextMove(moves, scores, i);
            int m = moves.moves[i];
            if (m != forbiddenMove && !detectThreeRepeats(pos, m)) return m;
        }

        // Last resort: return any move that differs from forbiddenMove, still in score order.
        for (int i = 0; i < moves.count; i++) {
            pickNextMove(moves, scores, i);
            int m = moves.moves[i];
            if (m != forbiddenMove) return m;
        }

        return forbiddenMove;
    }

    // =========================================================================
    // Guess table — populated by GuessingThread during the opponent's clock
    // =========================================================================

    static class GuessTable {

        private final int[] opponentGuesses = new int[Constants.MAX_GUESSES];
        private final int[] replyMoves      = new int[Constants.MAX_GUESSES];
        private volatile int size           = 0;
        private volatile boolean done       = false;

        // Returns our pre-computed reply for opponentMove, or 0 if not found.
        int lookupReply(int opponentMove) {
            for (int i = 0; i < size; i++)
                if (opponentGuesses[i] == opponentMove) return replyMoves[i];
            return 0;
        }

        void    finish() { done = true; }
        boolean isDone() { return done; }
    }

    // =========================================================================
    // GuessingThread — builds the guess table while the opponent thinks
    // =========================================================================

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
            MoveList oppPseudo     = new MoveList();
            MoveList opponentMoves = new MoveList();
            posAfterOurMove.legalMoves(oppPseudo, opponentMoves);
            if (opponentMoves.count == 0) { table.finish(); return; }

            // Score each opponent move from the opponent's perspective so the
            // most plausible replies are guessed first.
            int[] scores = new int[opponentMoves.count];
            for (int i = 0; i < opponentMoves.count; i++)
                scores[i] = -evaluate(posAfterOurMove.makeMove(opponentMoves.moves[i]));

            // Selection sort — descending by score.
            for (int i = 0; i < opponentMoves.count - 1; i++)
                for (int j = i + 1; j < opponentMoves.count; j++)
                    if (scores[j] > scores[i]) {
                        int t = opponentMoves.moves[i];
                        opponentMoves.moves[i] = opponentMoves.moves[j];
                        opponentMoves.moves[j] = t;
                        int s = scores[i]; scores[i] = scores[j]; scores[j] = s;
                    }

            // Stage 1: Fill the table — ask the NN for our best policy reply to each of the top opponent moves.
            int generated = 0;
            for (int i = 0; i < opponentMoves.count && !table.isDone() && generated < Constants.MAX_GUESSES; i++) {
                int opponentMove  = opponentMoves.moves[i];
                Position afterOpponent = posAfterOurMove.makeMove(opponentMove);

                MoveList ourPseudo = new MoveList();
                MoveList ourMoves  = new MoveList();
                afterOpponent.legalMoves(ourPseudo, ourMoves);
                if (ourMoves.count == 0) continue;

                int reply;
                try { reply = nn.topPolicyMove(afterOpponent, ourMoves); }
                catch (Exception e) { continue; }

                if (reply != 0) {
                    table.opponentGuesses[generated] = opponentMove;
                    table.replyMoves[generated]      = reply;
                    table.size                       = generated + 1;
                    generated++;
                }
                Thread.yield();
            }

            // Stage 2: Refine each stored reply with a shallow negamax search.
            // This replaces any tactically unsound NN suggestion with a better move.
            // Uses findBest (untimed) so it never interferes with the main search clock.
            for (int i = 0; i < table.size && !table.isDone(); i++) {
                Position afterOpponent = posAfterOurMove.makeMove(table.opponentGuesses[i]);
                int refined = findBest(afterOpponent, Constants.GUESS_REFINE_DEPTH, table.replyMoves[i], 0);
                if (refined != 0) table.replyMoves[i] = refined;
                Thread.yield();
            }

            table.finish();
        }
    }

    // =========================================================================
    // Opponent move detection and position equality
    // =========================================================================

    // Finds which legal move from `before` produced `after` by comparing board state.
    static int detectOpponentMove(Position before, Position after) {
        if (before == null || after == null) return 0;
        MoveList pseudo = new MoveList();
        MoveList moves  = new MoveList();
        before.legalMoves(pseudo, moves);
        for (int i = 0; i < moves.count; i++) {
            if (samePosition(before.makeMove(moves.moves[i]), after)) return moves.moves[i];
        }
        return 0;
    }

    // Compares all piece bitboards and game-state flags.
    static boolean samePosition(Position a, Position b) {
        return a.wp == b.wp && a.wn == b.wn && a.wb == b.wb &&
               a.wr == b.wr && a.wq == b.wq && a.wk == b.wk &&
               a.bp == b.bp && a.bn == b.bn && a.bb == b.bb &&
               a.br == b.br && a.bq == b.bq && a.bk == b.bk &&
               a.whiteToMove          == b.whiteToMove          &&
               a.whiteKingSideCastle  == b.whiteKingSideCastle  &&
               a.whiteQueenSideCastle == b.whiteQueenSideCastle &&
               a.blackKingSideCastle  == b.blackKingSideCastle  &&
               a.blackQueenSideCastle == b.blackQueenSideCastle;
    }
}
