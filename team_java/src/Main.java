
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Minimal UCI engine: plays the first legal move it finds.
 * Legal move generation is included (no castling, no en-passant; promotions -> queen only).
 */
public class Main {
    // sliding piece directions
    static final int[][] ROOK_DIRECTIONS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}; //right, left, up, down
    static final int[][] BISHOP_DIRECTIONS = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}; //rightUp, rightDown, leftUp, leftDown
    static final int[][] QUEEN_DIRECTIONS = {
            {1, 0}, {-1, 0}, {0, 1}, {0, -1},
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };

    /**
     * Main method that deals with communication and game logic.
     * Uses BufferedReader and PrintWriter to communicate over streams
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
            } else if (line.startsWith("go")) {
                List<Move> moves = pos.legalMoves();
                if (moves.isEmpty()) {
                    out.println("bestmove 0000");
                } else {
                    Move m = moves.get(0);
                    out.println("bestmove " + m.toUci());
                }
            } else if (line.equals("quit")) {
                break;
            }
            // ignore other UCI commands (setoption, etc.)
        }
        out.flush();
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
                Move m = Move.fromUci(uci);
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

    static Move iterativeDepthSearch(Position position) {
        startTime = System.currentTimeMillis();
        timeLimit = (long)(moveTimeMs * 0.85); // safety margin

        Move bestMove = null;

        for (int depth = 1; depth <= max_depth; depth++) {

            Move currentBest = findBestMove(position, depth, bestMove);

            if (System.currentTimeMillis() - startTime > timeLimit) {
                break;
            }

            bestMove = currentBest;
        }

        return bestMove;
    }


     /**
     * Explores the legal moves and finds the best move via min-max algorithm.
     * Uses alpha and beta values to prune branches and speed up evaluation time.
     * Stops recursion if time limit is reached.
     *
     * @return value of the moves at each recursion and the score of the board at base case.
     */
    static Move findBestMove(Position position, int depth, Move prevBest) {
    // Generate all legal moves from the current position
    List<Move> moves = position.legalMoves();

    // return null if no legal moves
    // Possible during checkmate or stalemate
    if (moves.isEmpty()) return null;

    
    orderMoves(position, moves);

    // If there was a best move from a previous search depth,
    // move it to the front so it gets searched first
  
    if (prevBest != null && moves.contains(prevBest)) {
        moves.remove(prevBest);
        moves.addFirst(prevBest);
    }

    
    Move bestMove = null; // tracks best move found so far

    // Initialize bestScore depending on whose turn it is:
    // White wants the highest score, Black wants the lowest score
    int bestScore = position.whiteToMove ? Integer.MIN_VALUE : Integer.MAX_VALUE;

    // Try each legal move
    for (Move m : moves) {
        // Stop searching if the allotted time has been exceeded
        if (System.currentTimeMillis() - startTime > timeLimit) break;

        
        Position next = position.makeMove(m); // get resulting position after making move

        // Evaluate the resulting position using alpha-beta search
        // depth - 1 because one move has already been made
        int score = alphaBeta(next, depth - 1, Integer.MIN_VALUE, Integer.MAX_VALUE, !position.whiteToMove);

        // If it's White's turn, choose the move with the highest score
        if (position.whiteToMove) {
            if (score > bestScore) {
                bestScore = score;
                bestMove = m;
            }
        } 
        // If it's Black's turn, choose the move with the lowest score
        else {
            if (score < bestScore) {
                bestScore = score;
                bestMove = m;
            }
        }
    }

    // Print the best move found at this depth for debugging
    if (bestMove != null) {
        System.out.println("Move: " + bestMove.toUci() + " Score: " + bestScore + " Depth: " + depth);
    }

    
    return bestMove; // returns best move
}
    static int alphaBeta(Position pos, int depth, int alpha, int beta, boolean maximizing) {
        List<Move> moves = pos.legalMoves();
        // base cases and time running out -> find score and return immediately
        if (System.currentTimeMillis() - startTime > timeLimit) return evaluate(pos);
        if (depth == 0) return evaluate(pos);
        // possible exception for checkmate or error
        if (moves.isEmpty()) return evaluate(pos);

        // pruning is faster when the best moves are at the top of the list for each layer searched
        orderMoves(pos, moves);

        if (maximizing) {
            int value = Integer.MIN_VALUE;
            for (Move m : moves) {
                value = Math.max(value,
                        alphaBeta(pos.makeMove(m), depth - 1, alpha, beta, false));
                alpha = Math.max(alpha, value);
                if (alpha >= beta) break; // prune
            }
            return value;
        } else {
            int value = Integer.MAX_VALUE;
            for (Move m : moves) {
                value = Math.min(value,
                        alphaBeta(pos.makeMove(m), depth - 1, alpha, beta, true));
                beta = Math.min(beta, value);
                if (beta <= alpha) break; // prune
            }
            return value;
        }
    }

    
    static void orderMoves(Position position, List<Move> moves) {
        
        moves.sort((a, b) -> {
            int scoreA = weightedMoveScore(position, a);
            int scoreB = weightedMoveScore(position, b);

            return Integer.compare(scoreB, scoreA); 
        });
    }

    static int weightedMoveScore(Position pos, Move move) {

        char target = pos.currentBoard[move.to];

        // prioritize captures
        if (target != '.') {
            return pieceValue(target) * 10;
        }
        // prioritize promotions
        if (move.promo != '0') {
            return 900;
        }
        return 0;
    }



    // ---------------- Chess core ----------------
    /* Rand & File to index table for real board
       File	a	b	c	d	e	f	g	h
       Rank
        8	56	57	58	59	60	61	62	63
        7	48	49	50	51	52	53	54	55
        6	40	41	42	43	44	45	46	47
        5	32	33	34	35	36	37	38	39
        4	24	25	26	27	28	29	30	31
        3	16	17	18	19	20	21	22	23
        2	8	9	10	11	12	13	14	15
        1	0	1	2	3	4	5	6	7
     */

    /**
     * Move class contains from index, to index and a character that determines piece promotion
     *
     */
    static int evaluate(Position pos) {
        int score = 0;

        for (int i = 0; i < 64; i++) {
            char p = pos.currentBoard[i];
            if (p == '.') continue;

            boolean isWhite = Character.isUpperCase(p);

            int base = pieceValue(p);
            int pst = 0;

            int idx = isWhite ? i : mirror(i);

            switch (Character.toUpperCase(p)) {
                case 'P': pst = PAWN_TABLE[idx]; break;
                case 'N': pst = KNIGHT_TABLE[idx]; break;
                case 'B': pst = BISHOP_TABLE[idx]; break;
                case 'R': pst = ROOK_TABLE[idx]; break;
                case 'Q': pst = QUEEN_TABLE[idx]; break;
                case 'K': pst = KING_TABLE[idx]; break;
            }

            int total = base + pst;

            score += isWhite ? total : -total;
        }

        return score;
    }

    static int mirror (int square) {
        return square ^ 56;
    }

    // simple method that assign each piece a value
    static int pieceValue(char p) {
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
    
    static final class Move {
        final int from; // 0..63
        final int to;   // 0..63
        final char promo; // 'q' or 0

        Move(int from, int to, char promo) {
            this.from = from;
            this.to = to;
            this.promo = promo;
        }

        // create a move object from Uci string
        static Move fromUci(String s) {
            if (s == null || s.length() < 4) return new Move(0, 0, (char) 0);
            int from = squareIndex(s.substring(0, 2));
            int to = squareIndex(s.substring(2, 4));
            char promo = (s.length() >= 5) ? s.charAt(4) : 0;
            return new Move(from, to, promo);
        }

        // create Uci string based on move object
        String toUci() {
            String u = indexToSquare(from) + indexToSquare(to);
            if (promo != 0) u += promo;
            return u;
        }

        // Helper conversion methods
        static int squareIndex(String sq) {
            int file = sq.charAt(0) - 'a';
            int rank = sq.charAt(1) - '1';
            return rank * 8 + file;
        }

        static String indexToSquare(int idx) {
            int file = idx % 8;
            int rank = idx / 8;
            return "" + (char) ('a' + file) + (char) ('1' + rank);
        }
    }

    // Class that uses fen format to keep track of moves internally
    // has a board and turn data
    /* Rand & File to index table for internal board
       File	0	1	2	3	4	5	6	7
       Rank
        7	56	57	58	59	60	61	62	63
        6	48	49	50	51	52	53	54	55
        5	40	41	42	43	44	45	46	47
        4	32	33	34	35	36	37	38	39
        3	24	25	26	27	28	29	30	31
        2	16	17	18	19	20	21	22	23
        1	8	9	10	11	12	13	14	15
        0	0	1	2	3	4	5	6	7
     */

    /**
     * Position class contains a character array represneting all 64 squares on the board
     * and which side it is to move
     */
    static final class Position {
        // board: '.' empty, pieces: PNBRQK for white, pnbrqk for black
        final char[] currentBoard = new char[64];
        // game states of each board
        final boolean whiteToMove;
        final boolean whiteKingSideCastle;
        final boolean whiteQueenSideCastle;
        final boolean blackKingSideCastle;
        final boolean blackQueenSideCastle;

        Position(char[] board, boolean wtm,
                 boolean wK, boolean wQ, boolean bK, boolean bQ) {
            System.arraycopy(board, 0, currentBoard, 0, 64);
            this.whiteToMove = wtm;
            this.whiteKingSideCastle = wK;
            this.whiteQueenSideCastle = wQ;
            this.blackKingSideCastle = bK;
            this.blackQueenSideCastle = bQ;
        }

        static void printBoardWithIndices(char[] board) {
            System.out.println();

            for (int rank = 7; rank >= 0; rank--) {
                System.out.print((rank + 1) + "  ");

                for (int file = 0; file < 8; file++) {
                    int index = rank * 8 + file;
                    char piece = board[index];

                    System.out.printf("%2s ", piece);
                }

                System.out.print("   ");

                // Print indices
                for (int file = 0; file < 8; file++) {
                    int index = rank * 8 + file;
                    System.out.printf("%2d ", index);
                }

                System.out.println();
            }

            System.out.println("\n   a  b  c  d  e  f  g  h\n");
        }


        // black at top of board, white at bottom of board
        static Position startPos() {
            return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        } // "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        /**
         * Creates a position object (board + turn) from fen string
         *
         * @param fen String that contains the FEN input command
         * @return new Postion object with board and turn set up as in the FEN string
         */
        static Position fromFEN(String fen) {
            // split string using spaces
            String[] fenTokens = fen.trim().split("\\s+");
            String placement = fenTokens[0];
            boolean wtm = fenTokens.length > 1 ? fenTokens[1].equals("w") : true;
            // initialize board with empty squares
            char[] board = new char[64];
            Arrays.fill(board, '.');

            // start at top left corner of the board
            int rank = 7;
            int file = 0;
            // loop through all 64 indexes and put the pieces in
            for (int i = 0; i < placement.length(); i++) {
                char c = placement.charAt(i);
                // decrease rank at /
                if (c == '/') {
                    rank--;
                    file = 0;
                // skip squares at numbers of free spaces
                } else if (Character.isDigit(c)) {
                    file += (c - '0');
                // place the characters at correct index
                } else {
                    int idx = rank * 8 + file;
                    board[idx] = c;
                    file++;
                }
            }
            boolean[] cRights = setCastlingRights(fenTokens[3]);
            // printBoardWithIndices(board);
            return new Position(board, wtm, cRights[0], cRights[1], cRights[2], cRights[3]);
        }

        /**
         * Swaps character between to and from, also checks and makes castling moves
         *
         * @param m Move object containing movement data used to update the board
         */
        Position makeMove(Move m) {
            char[] newBoard = Arrays.copyOf(currentBoard, 64);
            // store piece and put empty char in place
            char piece = newBoard[m.from];
            newBoard[m.from] = '.';
            char placed = piece;
            // Promotion logic for pawns
            if (m.promo != 0 && (piece == 'P' || piece == 'p')) {
                placed = (Character.isUpperCase(piece)) ? Character.toUpperCase(m.promo) : Character.toLowerCase(m.promo);
            }
            // put the piece at the to index
            newBoard[m.to] = placed;

            // Castling
            // copy castling rights of current board
            boolean wK = whiteKingSideCastle;
            boolean wQ = whiteQueenSideCastle;
            boolean bK = blackKingSideCastle;
            boolean bQ = blackQueenSideCastle;
            // if king moves remove castling rights
            if (piece == 'K') {
                wK = false;
                wQ = false;
            }
            if (piece == 'k') {
                bK = false;
                bQ = false;
            }
            // if rook moves for the first time, remove rights
            if (m.from == 0) wQ = false;
            if (m.from == 7) wK = false;
            if (m.from == 56) bQ = false;
            if (m.from == 63) bK = false;
            // if rook gets captured while in their initial square, remove rights
            if (m.to == 0) wQ = false;
            if (m.to == 7) wK = false;
            if (m.to == 56) bQ = false;
            if (m.to == 63) bK = false;
            // move the rook if a castling move occurs
            // white king-side
            if (piece == 'K' && m.from == 4 && m.to == 6) {
                newBoard[7] = '.';
                newBoard[5] = 'R';
            }
            // white queen-side
            if (piece == 'K' && m.from == 4 && m.to == 2) {
                newBoard[0] = '.';
                newBoard[3] = 'R';
            }
            // black king-side
            if (piece == 'k' && m.from == 60 && m.to == 62) {
                newBoard[63] = '.';
                newBoard[61] = 'r';
            }
            // black queen-side
            if (piece == 'k' && m.from == 60 && m.to == 58) {
                newBoard[56] = '.';
                newBoard[59] = 'r';
            }

            return new Position(newBoard, !whiteToMove, wK, wQ, bK, bQ);
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

        // goes through pseudo-legal moves to see if any move puts the King in check
        List<Move> legalMoves() {
            List<Move> out = new ArrayList<>();
            for (Move m : pseudoLegalMoves()) {
                // System.out.println(m.from + ", " + m.to + ", " + m.promo);
                // make the move to test it
                Position newPos = makeMove(m);
                // add the move if the move does not end in check of the King
                if (!newPos.inCheck(!newPos.whiteToMove)) {
                    out.add(m);
                }
            }
            return out;
        }

        // looks for King by looping through all squares and calls isSquareAttacked
        boolean inCheck(boolean whiteKing) {
            int kingSquare = -1;
            char king = whiteKing ? 'K' : 'k';
            // look for correct side King piece
            for (int i = 0; i < 64; i++)
                if (currentBoard[i] == king) {
                    kingSquare = i;
                    break;
                }
            if (kingSquare < 0) return true; // king missing => treat as in check
            return isSquareAttacked(kingSquare, !whiteKing);
        }

        /**
         * Determines whether a given square is attacked by the opponent.
         *
         * @param squareIndex Index (0–63) of the square being tested
         * @param attackedByWhite true if checking attacks from White, false for Black
         * @return true if the square is under attack, false otherwise
         */
        boolean isSquareAttacked(int squareIndex, boolean attackedByWhite) {
            // get rank and file of the square
            int rank = squareIndex / 8;
            int file = squareIndex % 8;

            // pawns
            if (attackedByWhite) {
                // ignore check if the pawn position is out of bounds
                // white pawns attack upward so check squares diagonally below square
                if (rank > 0 && file > 0 && currentBoard[(rank - 1) * 8 + (file - 1)] == 'P') return true;
                if (rank > 0 && file < 7 && currentBoard[(rank - 1) * 8 + (file + 1)] == 'P') return true;

            } else {
                // black pawns attack downward so check squares diagonally above square
                if (rank < 7 && file > 0 && currentBoard[(rank + 1) * 8 + (file - 1)] == 'p') return true;
                if (rank < 7 && file < 7 && currentBoard[(rank + 1) * 8 + (file + 1)] == 'p') return true;
            }

            // knights
            // knights have fixed offsets relative to squareIndex
            /* Example showcase of knightOffsets
               File	0	1	2	3	4	5	6	7
               Rank
                7	.	.	.	.	.	.	.	.
                6	.	.	.	.	.	.	.	.
                5	.	.	.	.	.	.	.	.
                4	.	.	15	.	17	.	.	.
                3	.	6	.	.	.	10	.	.
                2	.	.	-	Sq	+	.	.	.
                1	.  -10	.	.	.  -6	.	.
                0	.	.  -17	.  -15	.	.	.
             */
            int[] knightOffsets = {-17, -15, -10, -6, 6, 10, 15, 17};
            for (int offset : knightOffsets) {
                int targetIndex = squareIndex + offset;
                // Check board bounds, skips one iteration if out
                if (targetIndex < 0 || targetIndex >= 64) continue;
                // get possible knight's rank and file
                int targetRank = targetIndex / 8;
                int targetFile = targetIndex % 8;
                int rankDiff = Math.abs(targetRank - rank);
                int fileDiff = Math.abs(targetFile - file);
                /* Example showcase of warparound knightOffsets
                   File	0	1	2	3	4	5	6	7
                   Rank
                    7	.	.	.	.	.	.	.	.
                    6	.	.	.	.	.	.	.	.
                    5	.	.	.	.	.	.	.	.
                    4	.	17	.	.	.	.	.	.
                    3	.	.	10	.	.	.	.	15
                    2	Sq	+	.	.	.	.	6	.
                    1	.   .  -6	.	.   .	.	-
                    0	.  -15   .	.   .	.  -10	.
                    only 17, 10, -6, -15 are valid, -17 is out of bound, and 15, 6, -10 are warparounds
                */
                // Validate correct L-shape movement (prevents wraparound)
                if (!((rankDiff == 1 && fileDiff == 2) || (rankDiff == 2 && fileDiff == 1))) continue;
                // check the check at the correct offsets
                char piece = currentBoard[targetIndex];
                // if piece is a knight return true
                if (attackedByWhite && piece == 'N') return true;
                if (!attackedByWhite && piece == 'n') return true;
            }

            // Sliding Pieces: Rook, Bishop, Queen, and King
            // loop through all directions array
            for (int dirIndex = 0; dirIndex < QUEEN_DIRECTIONS.length; dirIndex++) {
                // get inner array and step in the direction from the square
                int fileStep = QUEEN_DIRECTIONS[dirIndex][0];
                int rankStep = QUEEN_DIRECTIONS[dirIndex][1];
                int currentRank = rank + rankStep;
                int currentFile = file + fileStep;

                // traverse outward in this direction as long as it is within bounds
                while (currentRank >= 0 && currentRank < 8 && currentFile >= 0 && currentFile < 8) {
                    // get piece index and char from board
                    int currentIndex = currentRank * 8 + currentFile;
                    char piece = currentBoard[currentIndex];
                    // keep stepping as long as it is empty
                    if (piece != '.') {
                        boolean isWhitePiece = Character.isUpperCase(piece);

                        // only consider opponent's pieces
                        if (isWhitePiece == attackedByWhite) {
                            // change to uppercase to simpler logic below
                            char pieceType = Character.toUpperCase(piece);
                            boolean isRookDirection = (dirIndex < 4);
                            boolean isBishopDirection = (dirIndex >= 4);
                            // queen (moves in all directions)
                            if (pieceType == 'Q') return true;
                            // rook (orthogonal)
                            if (isRookDirection && pieceType == 'R') return true;
                            // bishop (diagonal)
                            if (isBishopDirection && pieceType == 'B') return true;
                            // king (only valid if 1 square away)
                            // might not be necessary for inCheck since opponent's King cannot check the King
                            // but still useful to see if opponent's King can attack other pieces
                            if (pieceType == 'K' &&
                                    Math.max(Math.abs(currentRank - rank), Math.abs(currentFile - file)) == 1) {
                                return true;
                            }
                        }

                        // Stop scanning in this direction after hitting any piece
                        break;
                    }
                    // continue to next step in the direction of choice
                    currentRank += rankStep;
                    currentFile += fileStep;
                }
            }

            // redundant safety check for king attacks (covers edge cases)
            // loops through all 8 adjacent squares
            for (int r = rank - 1; r <= rank + 1; r++) {
                for (int f = file - 1; f <= file + 1; f++) {
                    // check for board bound violations
                    if (r < 0 || r >= 8 || f < 0 || f >= 8) continue;
                    // skips the middle square
                    if (r == rank && f == file) continue;

                    char piece = currentBoard[r * 8 + f];
                    if (attackedByWhite && piece == 'K') return true;
                    if (!attackedByWhite && piece == 'k') return true;
                }
            }

            return false;
        }

        /**
         * Generates all pseudo-legal moves for the current position.
         *
         * Pseudo-legal = follows piece movement rules ONLY.
         * Does NOT check for:
         *  - king in check
         *  - pinned pieces
         *  - illegal castling conditions
         *
         * @return List of all pseudo-legal moves for the side to move
         */
        List<Move> pseudoLegalMoves() {
            List<Move> moveList = new ArrayList<>();
            // iterate through all 64 squares
            for (int squareIndex = 0; squareIndex < 64; squareIndex++) {

                char piece = currentBoard[squareIndex];
                // skip empty squares
                if (piece == '.') continue;

                // determine piece color and generate for correct color
                boolean isWhitePiece = Character.isUpperCase(piece);
                if (isWhitePiece != whiteToMove) continue;

                // normalize piece type (uppercase for switch logic)
                char pieceType = Character.toUpperCase(piece);
                switch (pieceType) {
                    case 'P':
                        genPawn(moveList, squareIndex, isWhitePiece);
                        break;

                    case 'N':
                        genKnight(moveList, squareIndex, isWhitePiece);
                        break;

                    case 'B':
                        genSlidingPieces(moveList, squareIndex, isWhitePiece, BISHOP_DIRECTIONS);
                        break;

                    case 'R':
                        genSlidingPieces(moveList, squareIndex, isWhitePiece, ROOK_DIRECTIONS);
                        break;

                    case 'Q':
                        genSlidingPieces(moveList, squareIndex, isWhitePiece, QUEEN_DIRECTIONS);
                        break;

                    case 'K':
                        genKing(moveList, squareIndex, isWhitePiece);
                        break;
                }
            }

            return moveList;
        }

        void genPawn(List<Move> moveList, int fromSquare, boolean isWhite) {
            int rank = fromSquare / 8;
            int file = fromSquare % 8;

            int forwardStep = isWhite ? 8 : -8; //bc of array style to move forward we actually move 8 spaces in the array
            int startRank = isWhite ? 1 : 6; //the ranks on which pawns start
            int promotionRank = isWhite ? 6 : 1; //pawn cannot legally exit on last rank so we prep for promotion on second to last rank

            int oneForward = fromSquare + forwardStep; //space that is oneForward from current square

            if (oneForward >= 0 && oneForward < 64 && currentBoard[oneForward] == '.'){ //checks if the move is legal (if it stays on board and if it is on to an empty space)
                if (rank == promotionRank){
                    moveList.add(new Move(fromSquare, oneForward, 'q')); //promotion list
                    moveList.add(new Move(fromSquare, oneForward, 'n'));
                    moveList.add(new Move(fromSquare, oneForward, 'b'));
                    moveList.add(new Move(fromSquare, oneForward, 'r'));
                }
                else {
                    moveList.add(new Move(fromSquare, oneForward, (char) 0)); //else if its not promoting just move up one space and no promo
                }

                int twoForward = fromSquare + (2 * forwardStep); //calculates space in the array which would be two forward
                if (rank == startRank && twoForward >=0 && twoForward <64 && currentBoard[twoForward] == '.'){ //makes sure space is empty, in bounds, and pawn is on starting rank.
                    moveList.add(new Move(fromSquare, twoForward, (char) 0)); //if conditions are met you can move two spaces up.
                }
            }

            int [] captureFileSteps = {-1,1}; //diagonally would be up and to the left or to the right. thats what this is for.
            for (int fileStep : captureFileSteps) { //runs for every filestep in captureFileSteps
                int targetFile = file + fileStep; //to get the correct file for capturing
                if (targetFile <0 || targetFile > 7) continue; //if outside board bounds it will skip current iteration

                int targetRank = rank + (isWhite ? 1 : -1); //to get correct correct rank, (if its white it is 1 if not its -1 because the colors matter in which direction going)
                if (targetRank < 0 || targetRank >7) continue; //if out of bounds rank skip iteration of loop

                int toSquare = targetRank * 8 + targetFile; //this gives us a value in our 1D array for what square to go
                char targetPiece = currentBoard[toSquare]; //here we check what is there

                if (targetPiece == '.') continue; //if its empty continue/skip iteration since there is nothing to capture making it illegal to move there

                boolean isTargetWhite = Character.isUpperCase(targetPiece); //checks to see if piece is uppercase(white) or lowercase(black)
                if(isTargetWhite != isWhite){ //condition stating that if the piece colors are not the same then ....
                    if (rank == promotionRank){ //if rank is a promotion rank then lets add possible promotions
                        moveList.add(new Move(fromSquare, toSquare, 'q'));
                        moveList.add(new Move(fromSquare, toSquare, 'n'));
                        moveList.add(new Move(fromSquare, toSquare, 'b'));
                        moveList.add(new Move(fromSquare, toSquare, 'r'));//add more promos
                    }
                    else{
                        moveList.add(new Move(fromSquare, toSquare, (char) 0 )); //if its not a promotion rank then just capture and dont promote
                    }
                }
            }
        

        void genKnight(List<Move> moveList, int fromSquare, boolean isWhite) {
            int rank = fromSquare / 8;
            int file = fromSquare % 8;
        
            int[][] knightMoves = {
                    {1, 2}, {2, 1},
                    {2, -1}, {1, -2},
                    {-1, -2}, {-2, -1},
                    {-2, 1}, {-1, 2}
            };
        
            for (int[] move : knightMoves) {
                int targetFile = file + move[0];
                int targetRank = rank + move[1];
        
                if (targetFile < 0 || targetFile > 7 || targetRank < 0 || targetRank > 7) continue;
        
                int toSquare = targetRank * 8 + targetFile;
                char targetPiece = currentBoard[toSquare];
        
                if (targetPiece == '.') {
                    moveList.add(new Move(fromSquare, toSquare, (char) 0));
                    continue;
                }
        
                boolean isTargetWhite = Character.isUpperCase(targetPiece);
                if (isTargetWhite != isWhite) {
                    moveList.add(new Move(fromSquare, toSquare, (char) 0));
                }
            }

        }

        void genSlidingPieces(List<Move> moveList, int fromSquare, boolean isWhite, int[][] directions) {
            int rank = fromSquare / 8;
            int file = fromSquare % 8;

            for (int[] dir : directions) {
                int fileStep = dir[0];
                int rankStep = dir[1];

                int currentRank = rank + rankStep;
                int currentFile = file + fileStep;

                // Slide in this direction
                while (currentRank >= 0 && currentRank < 8 && currentFile >= 0 && currentFile < 8) {
                    int toSquare = currentRank * 8 + currentFile;
                    char piece = currentBoard[toSquare];

                    if (piece == '.') {
                        // Empty square => normal move
                        moveList.add(new Move(fromSquare, toSquare, (char) 0));
                    } else {
                        boolean isTargetWhite = Character.isUpperCase(piece);
                        if (isTargetWhite != isWhite) {
                            // Enemy piece => capture
                           moveList.add(new Move(fromSquare, toSquare, (char) 0));
                        }
                        // Stop sliding after hitting any piece
                        break;
                    }
                    currentRank += rankStep;
                    currentFile += fileStep;
                }
            }
        }

        void genKing(List<Move> moveList, int fromSquare, boolean isWhite) {
            // normal king moves...
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) {
                        continue;
                    }
                    int nr = r + dr;
                    int nf = f + df;

                    if (nr < 0 || nr >= 8 || nf < 0 || nf >= 8) {
                        continue;
                    }
                    int to = nr * 8 + nf;
                    char target = currentBoard[to];

                    // for empty square or an opponents piece
                    if (target == '.' || Character.isUpperCase(target) != white) {
                        ms.add(new Move(from, to, (char)0));
                    }
                }
            }
            // Castling
            // Redundant check to see if king is in the correct position to castle
            if (isWhite && fromSquare == 4 && currentBoard[4] == 'K') {
                // King-side (e1 → g1)
                if (whiteKingSideCastle && currentBoard[5] == '.' && currentBoard[6] == '.' &&
                        !isSquareAttacked(4, false) &&
                        !isSquareAttacked(5, false) &&
                        !isSquareAttacked(6, false) &&
                        currentBoard[7] == 'R') {
                    moveList.add(new Move(4, 6, '0')); // e1g1
                }
                // Queen-side (e1 → c1)
                if (whiteQueenSideCastle && currentBoard[1] == '.' && currentBoard[2] == '.' && currentBoard[3] == '.' &&
                        !isSquareAttacked(4, false) &&
                        !isSquareAttacked(3, false) &&
                        !isSquareAttacked(2, false) &&
                        currentBoard[0] == 'R') {
                    moveList.add(new Move(4, 2, '0')); // e1c1
                }
            }
            // For black king
            if (!isWhite && fromSquare == 60 && currentBoard[60] == 'k') {
                // Black king-side (e8 → g8)
                if (blackKingSideCastle && currentBoard[61] == '.' && currentBoard[62] == '.' &&
                        !isSquareAttacked(60, true) &&
                        !isSquareAttacked(61, true) &&
                        !isSquareAttacked(62, true) &&
                        currentBoard[63] == 'r') {
                    moveList.add(new Move(60, 62, '0')); // e8g8
                }
                // Black queen-side (e8 → c8)
                if (blackQueenSideCastle && currentBoard[57] == '.' && currentBoard[58] == '.' && currentBoard[59] == '.' &&
                        !isSquareAttacked(60, true) &&
                        !isSquareAttacked(59, true) &&
                        !isSquareAttacked(58, true) &&
                        currentBoard[56] == 'r') {
                    moveList.add(new Move(60, 58, '0')); // e8c8
                }
            }
        }
    }
}
