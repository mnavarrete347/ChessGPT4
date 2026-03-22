package annoucement;

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

    private static Position parsePosition(String cmd, Position current) {
        // UCI formats:
        // position startpos [moves ...]
        // position fen <fen...> [moves ...]
        String[] tokens = cmd.split("\\s+");
        int i = 1;
        Position p = current;

        //if the second token is "startpos"
        if (i < tokens.length && tokens[i].equals("startpos")) {
            p = Position.startPos();
            i++;
        } else if (i < tokens.length && tokens[i].equals("fen")) {
            i++;
            // FEN has 6 fields
            StringBuilder fen = new StringBuilder();
            for (int k = 0; k < 6 && i < tokens.length; k++, i++) {
                if (k > 0) fen.append(' ');
                fen.append(tokens[i]);
            }
            p = Position.fromFEN(fen.toString());
        }

        // if next token is moves
        if (i < tokens.length && tokens[i].equals("moves")) {
            i++;
            for (; i < tokens.length; i++) {
                String uci = tokens[i];
                Move m = Move.fromUci(uci);
                p = p.makeMove(m);
            }
        }
        return p;
    }

    // ---------------- Chess core ----------------

    static final class Move {
        final int from; // 0..63
        final int to;   // 0..63
        final char promo; // 'q' or 0

        Move(int from, int to, char promo) {
            this.from = from;
            this.to = to;
            this.promo = promo;
        }

        static Move fromUci(String s) {
            if (s == null || s.length() < 4) return new Move(0, 0, (char) 0);
            int from = squareIndex(s.substring(0, 2));
            int to = squareIndex(s.substring(2, 4));
            char promo = (s.length() >= 5) ? s.charAt(4) : 0;
            return new Move(from, to, promo);
        }

        String toUci() {
            String u = indexToSquare(from) + indexToSquare(to);
            if (promo != 0) u += promo;
            return u;
        }

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

        /* Rand & File to index table
Rank / File	a	b	c	d	e	f	g	h
        8	56	57	58	59	60	61	62	63
        7	48	49	50	51	52	53	54	55
        6	40	41	42	43	44	45	46	47
        5	32	33	34	35	36	37	38	39
        4	24	25	26	27	28	29	30	31
        3	16	17	18	19	20	21	22	23
        2	8	9	10	11	12	13	14	15
        1	0	1	2	3	4	5	6	7
         */
    }

    static final class Position {
        // board: '.' empty, pieces: PNBRQK for white, pnbrqk for black
        final char[] currentBoard = new char[64];
        final boolean whiteToMove;

        Position(char[] board, boolean wtm) {
            System.arraycopy(board, 0, currentBoard, 0, 64);
            this.whiteToMove = wtm;
        }

        static Position startPos() {
            return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
        }

        static Position fromFEN(String fen) {
            String[] f = fen.trim().split("\\s+");
            String placement = f[0];
            boolean wtm = f.length > 1 ? f[1].equals("w") : true;
            char[] board = new char[64];
            Arrays.fill(board, '.');

            int rank = 7;
            int file = 0;
            for (int i = 0; i < placement.length(); i++) {
                char c = placement.charAt(i);
                if (c == '/') {
                    rank--;
                    file = 0;
                } else if (Character.isDigit(c)) {
                    file += (c - '0');
                } else {
                    int idx = rank * 8 + file;
                    board[idx] = c;
                    file++;
                }
            }
            return new Position(board, wtm);
        }

        Position makeMove(Move m) {
            char[] nb = Arrays.copyOf(currentBoard, 64);
            char piece = nb[m.from];
            nb[m.from] = '.';
            char placed = piece;
            if (m.promo != 0 && (piece == 'P' || piece == 'p')) {
                placed = (Character.isUpperCase(piece)) ? Character.toUpperCase(m.promo) : Character.toLowerCase(m.promo);
            }
            nb[m.to] = placed;
            return new Position(nb, !whiteToMove);
        }

        List<Move> legalMoves() {
            List<Move> out = new ArrayList<>();
            for (Move m : pseudoLegalMoves()) {
                Position np = makeMove(m);
                if (!np.inCheck(!np.whiteToMove)) { // after new move, side who just moved is !np.whiteToMove
                    out.add(m);
                }
            }
            return out;
        }

        boolean inCheck(boolean whiteKing) {
            int kingSquare = -1;
            char king = whiteKing ? 'K' : 'k';
            for (int i = 0; i < 64; i++)
                if (currentBoard[i] == king) {
                    kingSquare = i;
                    break;
                }
            if (kingSquare < 0) return true; // king missing => treat as in check
            return isSquareAttacked(kingSquare, !whiteKing);
        }

        boolean isSquareAttacked(int sq, boolean byWhite) {
            // pawns
            int r = sq / 8, f = sq % 8;
            if (byWhite) {
                if (r > 0 && f > 0 && currentBoard[(r - 1) * 8 + (f - 1)] == 'P') return true;
                if (r > 0 && f < 7 && currentBoard[(r - 1) * 8 + (f + 1)] == 'P') return true;
            } else {
                if (r < 7 && f > 0 && currentBoard[(r + 1) * 8 + (f - 1)] == 'p') return true;
                if (r < 7 && f < 7 && currentBoard[(r + 1) * 8 + (f + 1)] == 'p') return true;
            }

            // knights
            int[] nD = {-17, -15, -10, -6, 6, 10, 15, 17};
            for (int d : nD) {
                int to = sq + d;
                if (to < 0 || to >= 64) continue;
                int tr = to / 8, tf = to % 8;
                int dr = Math.abs(tr - r), df = Math.abs(tf - f);
                if (!((dr == 1 && df == 2) || (dr == 2 && df == 1))) continue;
                char p = currentBoard[to];
                if (byWhite && p == 'N') return true;
                if (!byWhite && p == 'n') return true;
            }

            // sliders: bishop/rook/queen
            int[][] dirs = {
                    {1, 0}, {-1, 0}, {0, 1}, {0, -1}, // rook
                    {1, 1}, {1, -1}, {-1, 1}, {-1, -1} // bishop
            };
            for (int di = 0; di < dirs.length; di++) {
                int dr = dirs[di][1];
                int df = dirs[di][0];
                int cr = r + dr, cf = f + df;
                while (cr >= 0 && cr < 8 && cf >= 0 && cf < 8) {
                    int idx = cr * 8 + cf;
                    char p = currentBoard[idx];
                    if (p != '.') {
                        boolean isW = Character.isUpperCase(p);
                        if (isW == byWhite) {
                            char up = Character.toUpperCase(p);
                            boolean rookDir = (di < 4);
                            boolean bishopDir = (di >= 4);
                            if (up == 'Q') return true;
                            if (rookDir && up == 'R') return true;
                            if (bishopDir && up == 'B') return true;
                            // king one-step
                            if (up == 'K' && Math.max(Math.abs(cr - r), Math.abs(cf - f)) == 1) return true;
                        }
                        break;
                    }
                    cr += dr;
                    cf += df;
                }
            }

            // king adjacency already covered by sliders on first step; also cover direct king in knight/pawn misses
            for (int rr = r - 1; rr <= r + 1; rr++) {
                for (int ff = f - 1; ff <= f + 1; ff++) {
                    if (rr < 0 || rr >= 8 || ff < 0 || ff >= 8 || (rr == r && ff == f)) continue;
                    char p = currentBoard[rr * 8 + ff];
                    if (byWhite && p == 'K') return true;
                    if (!byWhite && p == 'k') return true;
                }
            }

            return false;
        }

        List<Move> pseudoLegalMoves() {
            List<Move> ms = new ArrayList<>();

            for (int i = 0; i < 64; i++) {
                char p = currentBoard[i];
                if (p == '.') continue;
                boolean isW = Character.isUpperCase(p);
                if (isW != whiteToMove) continue;
                char up = Character.toUpperCase(p);
                switch (up) {
                    case 'P':
                        genPawn(ms, i, isW);
                        break;
                    case 'N':
                        genKnight(ms, i, isW);
                        break;
                    case 'B':
                        genBishop(ms, i, isW, new int[][]{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}});
                        break;
                    case 'R':
                        genRook(ms, i, isW, new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}});
                        break;
                    case 'Q':
                        genQueen(ms, i, isW, new int[][]{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}});
                        break;
                    case 'K':
                        genKing(ms, i, isW);
                        break;
                }
            }
            return ms;
        }

        void genPawn(List<Move> ms, int from, boolean white) {

        }

        void genKnight(List<Move> ms, int from, boolean white) {

        }

        void genBishop(List<Move> ms, int from, boolean white, int[][] dirs) {

        }

        void genRook(List<Move> ms, int from, boolean white, int[][] dirs) {

        }

        void genQueen(List<Move> ms, int from, boolean white, int[][] dirs) {

        }

        void genKing(List<Move> ms, int from, boolean white) {

        }
    }
}
