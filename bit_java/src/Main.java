
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Arrays;

/**
 * UCI engine: uses alpha-beta pruning to find bestmove.
 * Legal move generation, promotion, and castling are included (no en-passant).
 */
public class Main {
    public static final boolean ENABLE_EN_PASSANT = false;
    // piece offsets and tables
    static final int[] ROOK_OFFSETS = {8, -8, 1, -1};
    static final int[] BISHOP_OFFSETS = {7, -7, 9, -9};
    static final int[] QUEEN_OFFSETS = {8, -8, 1, -1, 7, -7, 9, -9};
    static final long FILE_A = 0x0101010101010101L;
    static final long FILE_H = 0x8080808080808080L;
    static final long RANK_3 = 0x0000000000FF0000L;
    static final long RANK_6 = 0x0000FF0000000000L;
    static final long RANK_8 = 0xFF00000000000000L;
    static final long RANK_1 = 0x00000000000000FFL;

    // White Piece-Square Tables (Corrected for 0=a1 indexing)
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

    static final long[] KNIGHT_ATTACKS = new long[64];
    static final long[] KING_ATTACKS = new long[64];

    static {
        for (int i = 0; i < 64; i++) {
            int r = i / 8;
            int f = i % 8;

            // --- Knight Attack Precomputation ---
            // The 8 possible L-shapes
            int[] drN = {-2, -2, -1, -1, 1, 1, 2, 2};
            int[] dfN = {-1, 1, -2, 2, -2, 2, -1, 1};
            long nMask = 0L;

            for (int j = 0; j < 8; j++) {
                int nr = r + drN[j];
                int nf = f + dfN[j];

                // Check if the destination is on the 8x8 board
                if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                    nMask |= (1L << (nr * 8 + nf));
                }
            }
            KNIGHT_ATTACKS[i] = nMask;

            // --- King Attack Precomputation ---
            // All 8 adjacent squares
            long kMask = 0L;
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) continue; // Skip the square the king is actually on

                    int nr = r + dr;
                    int nf = f + df;

                    if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                        kMask |= (1L << (nr * 8 + nf));
                    }
                }
            }
            KING_ATTACKS[i] = kMask;
        }
    }

    // Parameters for move searching
    static int max_depth = 100;
    static long startTime;
    static long timeLimit;
    static long moveTimeMs = 10000; // 10 sec default time

    /**
     * Main method that deals with communication and game logic.
     * Uses BufferedReader and PrintWriter to communicate over character streams
     *
     */
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);

        Position pos = Position.startPos();

        String line;
        while ((line = br.readLine()) != null) {
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
                int m = iterativeSearch(pos);
                if (m == 0) out.println("bestmove 0000");
                else out.println("bestmove " + Move.toUci(m));
                pos = pos.makeMove(m);
                pos.printBoard();
            } else if (line.startsWith("perft")) {
                runPerft(pos, 1);
                runPerft(pos, 2);
                runPerft(pos, 3);
                runPerft(pos, 4);
                runPerft(pos, 5);
                runPerft(pos, 6);
            } else if (line.equals("quit")) {
                break;
            }
            // ignore other UCI commands (setoption, etc.)
        }
        out.flush();
    }

    public static void runPerft(Position pos, int depth) {
        long start = System.nanoTime();
        long nodes = perft(pos, depth);
        long end = System.nanoTime();

        double durationSeconds = (end - start) / 1_000_000_000.0;
        long nps = (long) (nodes / durationSeconds);

        System.out.printf("Depth %d: %d nodes in %.3f seconds (NPS: %d)%n", depth, nodes, durationSeconds, nps);
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

    /**
     * Parses the position UCI command into tokens which are processed step by step
     *
     * @param cmd String that directly comes from UCI chess arena
     * @param currentPos Position object that contains the previous state of the internal chess board
     * @return pos A new position object with the updated board after making moves
     */
    private static Position parsePosition(String cmd, Position currentPos) {
        // UCI formats:
        // position startpos [moves ...]
        // position fen <fen...> [moves ...]
        String[] tokens = cmd.split("\\s+"); // split command using space
        int i = 1;
        Position pos = currentPos;

        // if the second token is "startpos" -> UCI format
        if (i < tokens.length && tokens[i].equals("startpos")) {
            pos = Position.startPos();
            i++;
            // second token is "fen" -> FEN format
        } else if (i < tokens.length && tokens[i].equals("fen")) {
            i++;
            // FEN has 6 fields
            StringBuilder fen = new StringBuilder();
            StringBuilder fenCastling = new StringBuilder();
            // remake the fen string
            for (int k = 0; k < 6 && i < tokens.length; k++, i++) {
                if (k > 0) fen.append(' ');
                fen.append(tokens[i]);
                if (k == 2)  fenCastling.append(tokens[i]);
            }
            // pass fen string to make board position
            pos = Position.fromFEN(fen.toString());
        }

        // if third token is "moves"
        if (i < tokens.length && tokens[i].equals("moves")) {
            i++; // go to fourth token
            for (; i < tokens.length; i++) {
                // create moves from uci string tokens
                String uci = tokens[i];
                int m = Move.fromUci(uci);
                pos = pos.makeMove(m);
            }
        }
        return pos;
    }

    static void parseGo(String cmd) {
        String[] tokens = cmd.trim().split("\\s+");

        // Example inputs:
        // "go"
        // "go movetime 5000"
        // "go depth 5"

        for (int i = 1; i < tokens.length; i++) {

            switch (tokens[i].toLowerCase()) {
                case "movetime":
                    if (i + 1 < tokens.length) {
                        moveTimeMs = Integer.parseInt(tokens[i + 1]);
                        i++; // skip value
                    }
                    break;

                case "depth":
                    if (i + 1 < tokens.length) {
                        max_depth = Integer.parseInt(tokens[i + 1]);
                        i++; // skip value
                    }
                    break;

                default:
                    moveTimeMs = 10000;
                    max_depth = 100;
                    break;
            }
        }
    }

    static int iterativeSearch(Position pos) {
        startTime = System.currentTimeMillis();
        timeLimit = (long)(moveTimeMs * 0.85);
        int bestMoveFound = 0;

        for (int depth = 1; depth <= max_depth; depth++) {
            int m = findBest(pos, depth);
            if (System.currentTimeMillis() - startTime > timeLimit) break;
            bestMoveFound = m;
        }
        return bestMoveFound;
    }

    static int findBest(Position pos, int depth) {
        MoveList moves = pos.legalMoves();
        if (moves.count == 0) return 0;

        orderMoves(pos, moves);

        int bestMove = moves.moves[0];
        int alpha = -1000000;
        int beta = 1000000;

        for (int i = 0; i < moves.count; i++) {
            int m = moves.moves[i];
            Position next = pos.makeMove(m);
            // We negate the result and flip alpha/beta because it's the opponent's turn
            int score = -negamax(next, depth - 1, -beta, -alpha);

            if (score > alpha) {
                alpha = score;
                bestMove = m;
            }
        }
        return bestMove;
    }

    static int negamax(Position pos, int depth, int alpha, int beta) {
        if (depth == 0) return evaluate(pos);
        if (System.currentTimeMillis() - startTime > timeLimit) return evaluate(pos);

        MoveList moves = pos.pseudoLegalMoves(); // We'll filter legality inside
        int legalCount = 0;

        // Quick Move Ordering (See section 3)
        orderMoves(pos, moves);

        for (int i = 0; i < moves.count; i++) {
            Position next = pos.makeMove(moves.moves[i]);
            if (next.inCheck(!next.whiteToMove)) continue; // Basic legality check
            legalCount++;

            int score = -negamax(next, depth - 1, -beta, -alpha);

            if (score >= beta) return beta; // Cutoff
            if (score > alpha) alpha = score;
        }

        if (legalCount == 0) {
            return pos.inCheck(pos.whiteToMove) ? -100000 + depth : 0; // Mate or Stalemate
        }

        return alpha;
    }

    static int evaluate(Position pos) {
        int score = 0;

        // 1. Material (using bitCount)
        score += 100 * (Long.bitCount(pos.wp) - Long.bitCount(pos.bp));
        score += 320 * (Long.bitCount(pos.wn) - Long.bitCount(pos.bn));
        score += 330 * (Long.bitCount(pos.wb) - Long.bitCount(pos.bb));
        score += 500 * (Long.bitCount(pos.wr) - Long.bitCount(pos.br));
        score += 900 * (Long.bitCount(pos.wq) - Long.bitCount(pos.bq));

        // 2. Positional (PST)
        // Be careful here: wp for Pawns, wn for Knights, wb for Bishops, etc.
        score += evalPST(pos.wp, PAWN_PST, true);
        score -= evalPST(pos.bp, PAWN_PST, false);

        score += evalPST(pos.wn, KNIGHT_PST, true);
        score -= evalPST(pos.bn, KNIGHT_PST, false);

        score += evalPST(pos.wb, BISHOP_PST, true);
        score -= evalPST(pos.bb, BISHOP_PST, false);

        score += evalPST(pos.wr, ROOK_PST, true);
        score -= evalPST(pos.br, ROOK_PST, false);

        score += evalPST(pos.wq, QUEEN_PST, true);
        score -= evalPST(pos.bq, QUEEN_PST, false);

        score += evalPST(pos.wk, KING_PST, true);
        score -= evalPST(pos.bk, KING_PST, false);

        return pos.whiteToMove ? score : -score;
    }

    private static int evalPST(long bitboard, int[] table, boolean isWhite) {
        int pstSum = 0;
        while (bitboard != 0) {
            int sq = Long.numberOfTrailingZeros(bitboard);
            // Mirror the square for black (flip vertically)
            pstSum += table[isWhite ? sq : (sq ^ 56)];
            bitboard &= (bitboard - 1);
        }
        return pstSum;
    }

    static void orderMoves(Position pos, MoveList list) {
        int[] scores = new int[list.count];
        for (int i = 0; i < list.count; i++) {
            scores[i] = scoreMove(pos, list.moves[i]);
        }

        // Sort moves based on scores (Simple Selection Sort for small lists)
        for (int i = 0; i < list.count - 1; i++) {
            for (int j = i + 1; j < list.count; j++) {
                if (scores[j] > scores[i]) {
                    int tempMove = list.moves[i];
                    list.moves[i] = list.moves[j];
                    list.moves[j] = tempMove;

                    int tempScore = scores[i];
                    scores[i] = scores[j];
                    scores[j] = tempScore;
                }
            }
        }
    }

    static int scoreMove(Position pos, int m) {
        int score = 0;
        int from = Move.getFrom(m);
        int to = Move.getTo(m);
        long toBit = 1L << to;

        // MVV-LVA: Prioritize captures of high-value pieces with low-value pieces
        if ((pos.allPieces & toBit) != 0) {
            int victimValue = getPieceValue(pos.getPieceAt(to));
            int attackerValue = getPieceValue(pos.getPieceAt(from));
            score = 1000 + (victimValue - (attackerValue / 10));
        }

        // Prioritize promotions
        if (Move.getPromo(m) != 0) score += 900;

        return score;
    }

    private static int getPieceValue(char p) {
        return switch (Character.toUpperCase(p)) {
            case 'P' -> 100;
            case 'N' -> 300;
            case 'B' -> 300;
            case 'R' -> 500;
            case 'Q' -> 900;
            case 'K' -> 10000;
            default -> 0;
        };
    }

    /**
     * Utility class to handle moves as primitive ints.
     * Bits 0-5: From Square (0-63)
     * Bits 6-11: To Square (0-63)
     * Bits 12-14: Promotion Piece (0:None, 1:q, 2:r, 3:b, 4:n)
     */
    static final class Move {
        public static int create(int from, int to, int promo) {
            return from | (to << 6) | (promo << 12);
        }

        public static int getFrom(int move) { return move & 0x3F; }
        public static int getTo(int move) { return (move >> 6) & 0x3F; }
        public static int getPromo(int move) { return (move >> 12) & 0x7; }

        static int fromUci(String s) {
            if (s == null || s.length() < 4) return 0;
            int from = squareIndex(s.substring(0, 2));
            int to = squareIndex(s.substring(2, 4));
            int promo = 0;
            if (s.length() >= 5) {
                char p = s.charAt(4);
                if (p == 'q') promo = 1;
                else if (p == 'r') promo = 2;
                else if (p == 'b') promo = 3;
                else if (p == 'n') promo = 4;
            }
            return create(from, to, promo);
        }

        static String toUci(int move) {
            String s = indexToSquare(getFrom(move)) + indexToSquare(getTo(move));
            int p = getPromo(move);
            if (p == 1) s += "q";
            else if (p == 2) s += "r";
            else if (p == 3) s += "b";
            else if (p == 4) s += "n";
            return s;
        }

        static int squareIndex(String sq) {
            return (sq.charAt(1) - '1') * 8 + (sq.charAt(0) - 'a');
        }

        static String indexToSquare(int idx) {
            return "" + (char) ('a' + (idx % 8)) + (char) ('1' + (idx / 8));
        }
    }

    static final class MoveList {
        // A square can't usually have more than 218 moves in any legal position
        public int[] moves = new int[256];
        public int count = 0;

        public void add(int move) {
            moves[count++] = move;
        }
    }

    /**
     * Position class contains a character array represneting all 64 squares on the board
     * and which side it is to move
     */

    static final class Position {
        // 12 Piece bitboards
        long wp, wn, wb, wr, wq, wk; // White: pawn, knight, bishop, rook, queen, king
        long bp, bn, bb, br, bq, bk; // Black: pawn, knight, bishop, rook, queen, king

        // Occupancy bitboards
        long whitePieces, blackPieces, allPieces;

        // Game State
        final boolean whiteToMove;
        final boolean whiteKingSideCastle, whiteQueenSideCastle;
        final boolean blackKingSideCastle, blackQueenSideCastle;
        final int enPassantSq;

        Position(long[] pieces, boolean wtm, boolean wK, boolean wQ, boolean bK, boolean bQ, int epSq) {
            this.wp = pieces[0]; this.wn = pieces[1]; this.wb = pieces[2];
            this.wr = pieces[3]; this.wq = pieces[4]; this.wk = pieces[5];
            this.bp = pieces[6]; this.bn = pieces[7]; this.bb = pieces[8];
            this.br = pieces[9]; this.bq = pieces[10]; this.bk = pieces[11];

            this.whitePieces = wp | wn | wb | wr | wq | wk;
            this.blackPieces = bp | bn | bb | br | bq | bk;
            this.allPieces = whitePieces | blackPieces;

            this.whiteToMove = wtm;
            this.whiteKingSideCastle = wK;
            this.whiteQueenSideCastle = wQ;
            this.blackKingSideCastle = bK;
            this.blackQueenSideCastle = bQ;
            this.enPassantSq = epSq;
        }

        // black at top of board, white at bottom of board
        static Position startPos() {
            return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        } // "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        static Position fromFEN(String fen) {
            String[] fenTokens = fen.trim().split("\\s+");
            String placement = fenTokens[0];
            boolean wtm = fenTokens.length > 1 ? fenTokens[1].equals("w") : true;

            long[] pieces = new long[12];
            int rank = 7, file = 0;

            for (int i = 0; i < placement.length(); i++) {
                char c = placement.charAt(i);
                if (c == '/') { rank--; file = 0; }
                else if (Character.isDigit(c)) { file += (c - '0'); }
                else {
                    int sq = rank * 8 + file;
                    int pIndex = getPieceIndex(c);
                    pieces[pIndex] |= (1L << sq); // Set the bit at square index
                    file++;
                }
            }

            boolean[] cRights = setCastlingRights(fenTokens.length > 2 ? fenTokens[2] : "-");
            String epToken = fenTokens.length > 3 ? fenTokens[3] : "-";
            int epSq = epToken.equals("-") ? -1 : Move.squareIndex(epToken);

            return new Position(pieces, wtm, cRights[0], cRights[1], cRights[2], cRights[3], epSq);
        }

        private static int getPieceIndex(char c) {
            return switch (c) {
                case 'P' -> 0; case 'N' -> 1; case 'B' -> 2; case 'R' -> 3; case 'Q' -> 4; case 'K' -> 5;
                case 'p' -> 6; case 'n' -> 7; case 'b' -> 8; case 'r' -> 9; case 'q' -> 10; case 'k' -> 11;
                default -> -1;
            };
        }

        // Helper to get piece at a square (useful for debugging/UI)
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

        void printBoard() {
            System.out.println();
            for (int rank = 7; rank >= 0; rank--) {
                System.out.print((rank + 1) + "  "); // Rank labels (8-1)
                for (int file = 0; file < 8; file++) {
                    int sq = rank * 8 + file;
                    System.out.print(getPieceAt(sq) + " ");
                }
                System.out.println();
            }
            System.out.println("\n   a b c d e f g h"); // File labels (a-h)
            System.out.println();
        }

        Position makeMove(int m) {
            // 1. Setup new piece bitboards
            long[] nextPieces = {wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk};

            int from = Move.getFrom(m);
            int to = Move.getTo(m);
            int promo = Move.getPromo(m);
            long fromBit = 1L << from;
            long toBit = 1L << to;
            long moveMask = fromBit | toBit;

            // 2. Identify the moving piece and its color
            int movingPieceIdx = -1;
            for (int i = 0; i < 12; i++) {
                if ((nextPieces[i] & fromBit) != 0) {
                    movingPieceIdx = i;
                    break;
                }
            }

            // 3. Handle Captures (Regular)
            if ((allPieces & toBit) != 0) {
                // Find and remove the captured piece from its bitboard
                for (int i = 0; i < 12; i++) {
                    if ((nextPieces[i] & toBit) != 0) {
                        nextPieces[i] ^= toBit;
                        break;
                    }
                }
            }

            // 4. Handle En Passant Capture
            int nextEpSq = -1;
            if (ENABLE_EN_PASSANT && (movingPieceIdx == 0 || movingPieceIdx == 6) && to == enPassantSq) {
                int capturedPawnSq = whiteToMove ? (to - 8) : (to + 8);
                nextPieces[whiteToMove ? 6 : 0] ^= (1L << capturedPawnSq); // Remove enemy pawn
            }

            // 5. Move the piece
            nextPieces[movingPieceIdx] ^= moveMask;

            // 6. Handle Promotion
            if (promo != 0 && (movingPieceIdx == 0 || movingPieceIdx == 6)) {
                nextPieces[movingPieceIdx] ^= toBit; // Remove pawn from 'to' square
                int promoIdx = (whiteToMove ? 0 : 6) + (promo == 1 ? 4 : promo == 4 ? 1 : promo == 2 ? 3 : 2);
                // Note: Mapping our promo codes (1:q, 2:r, 3:b, 4:n) to our 12-index array
                int actualPromoIdx = whiteToMove ? (promo == 1 ? 4 : promo == 2 ? 3 : promo == 3 ? 2 : 1)
                        : (promo == 1 ? 10 : promo == 2 ? 9 : promo == 3 ? 8 : 7);
                nextPieces[actualPromoIdx] |= toBit;
            }

            // 7. Handle Double Pawn Push (Set next EP Square)
            if ((movingPieceIdx == 0 || movingPieceIdx == 6) && Math.abs(to - from) == 16) {
                nextEpSq = (from + to) / 2;
            }

            // 8. Handle Castling Rights & Rook Movement
            boolean wK = whiteKingSideCastle, wQ = whiteQueenSideCastle;
            boolean bK = blackKingSideCastle, bQ = blackQueenSideCastle;

            // If king or rook moves, or rook is captured, update rights
            if (from == 4 || to == 4) { wK = false; wQ = false; }
            if (from == 60 || to == 60) { bK = false; bQ = false; }
            if (from == 0 || to == 0) wQ = false;
            if (from == 7 || to == 7) wK = false;
            if (from == 56 || to == 56) bQ = false;
            if (from == 63 || to == 63) bK = false;

            // Move Rooks for Castling
            if (movingPieceIdx == 5) { // White King
                if (from == 4 && to == 6) nextPieces[3] ^= (1L << 7 | 1L << 5); // h1 to f1
                if (from == 4 && to == 2) nextPieces[3] ^= (1L << 0 | 1L << 3); // a1 to d1
            } else if (movingPieceIdx == 11) { // Black King
                if (from == 60 && to == 62) nextPieces[9] ^= (1L << 63 | 1L << 61); // h8 to f8
                if (from == 60 && to == 58) nextPieces[9] ^= (1L << 56 | 1L << 59); // a8 to d8
            }

            return new Position(nextPieces, !whiteToMove, wK, wQ, bK, bQ, nextEpSq);
        }

        /**
         * Sets the game state flags according to FEN input command
         *
         * @param fenCastling String that contains the FEN input command
         */
        static boolean[] setCastlingRights(String fenCastling) {
            boolean wk = fenCastling.contains("K");
            boolean wQ = fenCastling.contains("Q");
            boolean bK = fenCastling.contains("k");
            boolean bQ = fenCastling.contains("q");
            return new boolean[]{wk, wQ, bK, bQ};
        }

        /**
         * Ensures that adding an offset doesn't "wrap around" the board edges.
         */
        private boolean isWrap(int from, int to, int offset) {
            // (x & 7) is exactly the same as (x % 8) for positive integers
            int fromFile = from & 7;
            int toFile = to & 7;
            int fileDiff = Math.abs(toFile - fromFile);

            // Vertical moves (offset 8) must stay in the same file
            if (Math.abs(offset) == 8) return fileDiff != 0;

            // Sliding/Diagonal moves must only move exactly 1 file per step
            return fileDiff != 1;
        }

        // goes through pseudo-legal moves to see if any move puts the King in check
        MoveList legalMoves() {
            MoveList out = new MoveList();
            MoveList pseudo = pseudoLegalMoves();

            for (int i = 0; i < pseudo.count; i++) {
                int m = pseudo.moves[i];
                Position nextPos = makeMove(m);

                // After we move, we check if OUR king is now in check.
                // !whiteToMove is the correct check here because makeMove toggles the turn.
                if (!nextPos.inCheck(!nextPos.whiteToMove)) {
                    out.add(m);
                }
            }
            return out;
        }

        /**
         * Finds the king square bitwise and checks for attackers.
         */
        boolean inCheck(boolean isWhiteKing) {
            long kingBoard = isWhiteKing ? wk : bk;

            // If there's no king (shouldn't happen in real games), we're effectively in check.
            if (kingBoard == 0) return true;

            // Long.numberOfTrailingZeros(long) is a CPU-level instruction that
            // instantly gives us the index (0-63) of the set bit.
            int kingSq = Long.numberOfTrailingZeros(kingBoard);

            // Check if the enemy (the side that ISN'T the king) attacks this square.
            return isSquareAttacked(kingSq, !isWhiteKing);
        }

        boolean isSquareAttacked(int sq, boolean attackedByWhite) {
            long bit = 1L << sq;

            // 1. Attacked by Pawns
            // If White attacks, the pawn must be below the square.
            if (attackedByWhite) {
                if (((bit >> 7) & wp & ~FILE_A) != 0) return true;
                if (((bit >> 9) & wp & ~FILE_H) != 0) return true;
            } else {
                if (((bit << 7) & bp & ~FILE_H) != 0) return true;
                if (((bit << 9) & bp & ~FILE_A) != 0) return true;
            }

            // 2. Attacked by Knights (Lookup Table)
            long enemyKnights = attackedByWhite ? wn : bn;
            if ((KNIGHT_ATTACKS[sq] & enemyKnights) != 0) return true;

            // 3. Attacked by Kings (Lookup Table)
            long enemyKing = attackedByWhite ? wk : bk;
            if ((KING_ATTACKS[sq] & enemyKing) != 0) return true;

            // 4. Attacked by Sliders (Rook, Bishop, Queen)
            // We reuse the slider logic but return true immediately upon a hit
            if (checkSliderAttack(sq, attackedByWhite)) return true;

            return false;
        }

        private boolean checkSliderAttack(int sq, boolean white) {
            long enemyRooks = white ? (wr | wq) : (br | bq);
            long enemyBishops = white ? (wb | wq) : (bb | bq);

            // Check Orthogonals (Rook/Queen)
            for (int offset : ROOK_OFFSETS) {
                int current = sq;
                while (true) {
                    int next = current + offset;
                    if (next < 0 || next >= 64 || isWrap(current, next, offset)) break;
                    long bit = 1L << next;
                    if ((enemyRooks & bit) != 0) return true;
                    if ((allPieces & bit) != 0) break; // Blocked by any piece
                    current = next;
                }
            }

            // Check Diagonals (Bishop/Queen)
            for (int offset : BISHOP_OFFSETS) {
                int current = sq;
                while (true) {
                    int next = current + offset;
                    if (next < 0 || next >= 64 || isWrap(current, next, offset)) break;
                    long bit = 1L << next;
                    if ((enemyBishops & bit) != 0) return true;
                    if ((allPieces & bit) != 0) break; // Blocked by any piece
                    current = next;
                }
            }
            return false;
        }

        MoveList pseudoLegalMoves() {
            MoveList moveList = new MoveList();

            if (whiteToMove) {
                genPawn(moveList, true);
                genKnight(moveList, true);
                genSlidingPieces(moveList, true, BISHOP_OFFSETS, wb);
                genSlidingPieces(moveList, true, ROOK_OFFSETS, wr);
                genSlidingPieces(moveList, true, QUEEN_OFFSETS, wq);
                genKing(moveList, true);
            } else {
                genPawn(moveList, false);
                genKnight(moveList, false);
                genSlidingPieces(moveList, false, BISHOP_OFFSETS, bb);
                genSlidingPieces(moveList, false, ROOK_OFFSETS, br);
                genSlidingPieces(moveList, false, QUEEN_OFFSETS, bq);
                genKing(moveList, false);
            }

            return moveList;
        }

        private void serializeMoves(MoveList moveList, int from, long destinations) {
            while (destinations != 0) {
                int to = Long.numberOfTrailingZeros(destinations);
                moveList.add(Move.create(from, to, 0));
                destinations &= (destinations - 1);
            }
        }

        private void serializePawnMoves(MoveList moveList, long destinations, int offset, boolean promotion) {
            while (destinations != 0) {
                int to = Long.numberOfTrailingZeros(destinations);
                int from = to - offset;
                if (promotion) {
                    moveList.add(Move.create(from, to, 1)); // Queen
                    moveList.add(Move.create(from, to, 4)); // Knight
                    moveList.add(Move.create(from, to, 2)); // Rook
                    moveList.add(Move.create(from, to, 3)); // Bishop
                } else {
                    moveList.add(Move.create(from, to, 0));
                }
                destinations &= (destinations - 1);
            }
        }

        void genPawn(MoveList moveList, boolean isWhite) {
            long enemies = isWhite ? blackPieces : whitePieces;
            long empty = ~allPieces;

            if (isWhite) {
                // Pushes
                long singlePush = (wp << 8) & empty;
                serializePawnMoves(moveList, singlePush & ~RANK_8, 8, false);
                serializePawnMoves(moveList, singlePush & RANK_8, 8, true);

                long doublePush = ((singlePush & RANK_3) << 8) & empty;
                serializePawnMoves(moveList, doublePush, 16, false);

                // Captures
                long capLeft = (wp << 7) & enemies & ~FILE_H;
                serializePawnMoves(moveList, capLeft & ~RANK_8, 7, false);
                serializePawnMoves(moveList, capLeft & RANK_8, 7, true);

                long capRight = (wp << 9) & enemies & ~FILE_A;
                serializePawnMoves(moveList, capRight & ~RANK_8, 9, false);
                serializePawnMoves(moveList, capRight & RANK_8, 9, true);

                // En Passant
                if (enPassantSq != -1) {
                    long epBit = (1L << enPassantSq);
                    if (((wp << 7) & epBit & ~FILE_H) != 0) moveList.add(Move.create(enPassantSq - 7, enPassantSq, 0));
                    if (((wp << 9) & epBit & ~FILE_A) != 0) moveList.add(Move.create(enPassantSq - 9, enPassantSq, 0));
                }
            } else {
                // Black logic is mirrored (using >> and RANK_1/RANK_6)
                long singlePush = (bp >> 8) & empty;
                serializePawnMoves(moveList, singlePush & ~RANK_1, -8, false);
                serializePawnMoves(moveList, singlePush & RANK_1, -8, true);

                long doublePush = ((singlePush & RANK_6) >> 8) & empty;
                serializePawnMoves(moveList, doublePush, -16, false);

                long capLeft = (bp >> 9) & enemies & ~FILE_H;
                serializePawnMoves(moveList, capLeft & ~RANK_1, -9, false);
                serializePawnMoves(moveList, capLeft & RANK_1, -9, true);

                long capRight = (bp >> 7) & enemies & ~FILE_A;
                serializePawnMoves(moveList, capRight & ~RANK_1, -7, false);
                serializePawnMoves(moveList, capRight & RANK_1, -7, true);

                if (enPassantSq != -1) {
                    long epBit = (1L << enPassantSq);
                    if (((bp >> 9) & epBit & ~FILE_H) != 0) moveList.add(Move.create(enPassantSq + 9, enPassantSq, 0));
                    if (((bp >> 7) & epBit & ~FILE_A) != 0) moveList.add(Move.create(enPassantSq + 7, enPassantSq, 0));
                }
            }
        }

        void genKnight(MoveList moveList, boolean isWhite) {
            long knights = isWhite ? wn : bn;
            long friendly = isWhite ? whitePieces : blackPieces;

            while (knights != 0) {
                int from = Long.numberOfTrailingZeros(knights);
                long moves = KNIGHT_ATTACKS[from] & ~friendly;
                serializeMoves(moveList, from, moves);
                knights &= (knights - 1);
            }
        }

        void genKing(MoveList moveList, boolean isWhite) {
            int from = Long.numberOfTrailingZeros(isWhite ? wk : bk);
            long friendly = isWhite ? whitePieces : blackPieces;
            long moves = KING_ATTACKS[from] & ~friendly;
            serializeMoves(moveList, from, moves);

            // Castling
            if (isWhite) {
                if (whiteKingSideCastle && (allPieces & 0x60L) == 0 && !isSquareAttacked(4, false) && !isSquareAttacked(5, false) && !isSquareAttacked(6, false))
                    moveList.add(Move.create(4, 6, 0));
                if (whiteQueenSideCastle && (allPieces & 0x0EL) == 0 && !isSquareAttacked(4, false) && !isSquareAttacked(3, false) && !isSquareAttacked(2, false))
                    moveList.add(Move.create(4, 2, 0));
            } else {
                if (blackKingSideCastle && (allPieces & 0x6000000000000000L) == 0 && !isSquareAttacked(60, true) && !isSquareAttacked(61, true) && !isSquareAttacked(62, true))
                    moveList.add(Move.create(60, 62, 0));
                if (blackQueenSideCastle && (allPieces & 0x0E00000000000000L) == 0 && !isSquareAttacked(60, true) && !isSquareAttacked(59, true) && !isSquareAttacked(58, true))
                    moveList.add(Move.create(60, 58, 0));
            }
        }

        void genSlidingPieces(MoveList moveList, boolean isWhite, int[] offsets, long bitboard) {
            long friendly = isWhite ? whitePieces : blackPieces;
            long enemy = isWhite ? blackPieces : whitePieces;

            while (bitboard != 0) {
                int from = Long.numberOfTrailingZeros(bitboard);
                for (int offset : offsets) {
                    int current = from;
                    while (true) {
                        int next = current + offset;
                        if (next < 0 || next >= 64 || isWrap(current, next, offset)) break;

                        long bit = (1L << next);
                        if ((friendly & bit) != 0) break; // Blocked by friend

                        moveList.add(Move.create(from, next, 0));
                        if ((enemy & bit) != 0) break; // Captured enemy and stop

                        current = next;
                    }
                }
                bitboard &= (bitboard - 1);
            }
        }
    }
}
