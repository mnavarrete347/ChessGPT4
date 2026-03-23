
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
    // state variables / game flags
    static boolean whiteKingSideCastle;
    static boolean whiteQueenSideCastle;
    static boolean blackKingSideCastle;
    static boolean blackQueenSideCastle;

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
            // remake the fen string
            for (int k = 0; k < 6 && i < tokens.length; k++, i++) {
                if (k > 0) fen.append(' ');
                fen.append(tokens[i]);
            }
            // pass fen string to make board position
            pos = Position.fromFEN(fen.toString());
            setCastlingRights(fen.toString());
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

    private static void setCastlingRights(String fenCastling) {
        whiteKingSideCastle  = fenCastling.contains("K");
        whiteQueenSideCastle = fenCastling.contains("Q");
        blackKingSideCastle  = fenCastling.contains("k");
        blackQueenSideCastle = fenCastling.contains("q");
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
    static final class Position {
        // board: '.' empty, pieces: PNBRQK for white, pnbrqk for black
        final char[] currentBoard = new char[64];
        final boolean whiteToMove;

        Position(char[] board, boolean wtm) {
            System.arraycopy(board, 0, currentBoard, 0, 64);
            this.whiteToMove = wtm;
        }

        // black at top of board, white at bottom of board
        static Position startPos() {
            return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
        }

        // create a position object (board + turn) from fen string
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
            return new Position(board, wtm);
        }

        // swaps character between to and from
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
            return new Position(newBoard, !whiteToMove);
        }

        // goes through pseudo-legal moves to see if any move puts the King in check
        List<Move> legalMoves() {
            List<Move> out = new ArrayList<>();
            for (Move m : pseudoLegalMoves()) {
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
                        genBishop(moveList, squareIndex, isWhitePiece);
                        break;

                    case 'R':
                        genRook(moveList, squareIndex, isWhitePiece);
                        break;

                    case 'Q':
                        genQueen(moveList, squareIndex, isWhitePiece);
                        break;

                    case 'K':
                        genKing(moveList, squareIndex, isWhitePiece);
                        break;
                }
            }

            return moveList;
        }

        void genPawn(List<Move> moveList, int fromSquare, boolean isWhite) {

        }

        void genKnight(List<Move> moveList, int fromSquare, boolean isWhite) {

        }

        void genBishop(List<Move> moveList, int fromSquare, boolean isWhite) {

        }

        void genRook(List<Move> moveList, int fromSquare, boolean isWhite) {
            int rank = fromSquare / 8;
            int file = fromSquare % 8;

            for (int[] dir : ROOK_DIRECTIONS) {
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
                        moveList.add(new Move(fromSquare, toSquare, '0'));
                    } else {
                        boolean isTargetWhite = Character.isUpperCase(piece);
                        if (isTargetWhite != isWhite) {
                            // Enemy piece => capture
                            moveList.add(new Move(fromSquare, toSquare, '0'));
                        }
                        // Stop sliding after hitting any piece
                        break;
                    }
                    currentRank += rankStep;
                    currentFile += fileStep;
                }
            }
        }

        void genQueen(List<Move> moveList, int fromSquare, boolean isWhite) {

        }

        void genKing(List<Move> moveList, int fromSquare, boolean isWhite) {
            // normal king moves...


            // Castling
            if (isWhite) {
                // King-side (e1 → g1)
                if (whiteKingSideCastle && currentBoard[5] == '.' && currentBoard[6] == '.' &&
                        !isSquareAttacked(4, false) &&
                        !isSquareAttacked(5, false) &&
                        !isSquareAttacked(6, false)) {
                    moveList.add(new Move(4, 6, '0')); // e1g1
                }
                // Queen-side (e1 → c1)
                if (whiteQueenSideCastle && currentBoard[1] == '.' && currentBoard[2] == '.' && currentBoard[3] == '.' &&
                        !isSquareAttacked(4, false) &&
                        !isSquareAttacked(3, false) &&
                        !isSquareAttacked(2, false)) {
                    moveList.add(new Move(4, 2, '0')); // e1c1
                }

            } else {
                // Black king-side (e8 → g8)
                if (blackKingSideCastle && currentBoard[61] == '.' && currentBoard[62] == '.' &&
                        !isSquareAttacked(60, true) &&
                        !isSquareAttacked(61, true) &&
                        !isSquareAttacked(62, true)) {
                    moveList.add(new Move(60, 62, '0')); // e8g8
                }

                // Black queen-side (e8 → c8)
                if (blackQueenSideCastle && currentBoard[57] == '.' && currentBoard[58] == '.' && currentBoard[59] == '.' &&
                        !isSquareAttacked(60, true) &&
                        !isSquareAttacked(59, true) &&
                        !isSquareAttacked(58, true)) {
                    moveList.add(new Move(60, 58, '0')); // e8c8
                }
            }
        }
    }
}
