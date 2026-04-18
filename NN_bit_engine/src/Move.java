// Packed int: bits 0-5 = from, 6-11 = to, 12-14 = promo (0=none,1=Q,2=R,3=B,4=N)
public final class Move {

    private Move() {}

    public static int create(int from, int to, int promo) {
        return from | (to << 6) | (promo << 12);
    }

    public static int getFrom (int move) { return  move        & 0x3F; }
    public static int getTo   (int move) { return (move >>  6) & 0x3F; }
    public static int getPromo(int move) { return (move >> 12) & 0x07; }

    public static int fromUci(String s) {
        if (s == null || s.length() < 4) return 0;
        int from  = squareIndex(s.substring(0, 2));
        int to    = squareIndex(s.substring(2, 4));
        int promo = 0;
        if (s.length() >= 5) {
            promo = switch (s.charAt(4)) {
                case 'q' -> 1; case 'r' -> 2; case 'b' -> 3; case 'n' -> 4; default -> 0;
            };
        }
        return create(from, to, promo);
    }

    public static String toUci(int move) {
        String s = indexToSquare(getFrom(move)) + indexToSquare(getTo(move));
        return switch (getPromo(move)) {
            case 1  -> s + "q"; case 2  -> s + "r";
            case 3  -> s + "b"; case 4  -> s + "n";
            default -> s;
        };
    }

    public static int squareIndex(String sq) {
        return (sq.charAt(1) - '1') * 8 + (sq.charAt(0) - 'a');
    }

    public static String indexToSquare(int idx) {
        return "" + (char)('a' + idx % 8) + (char)('1' + idx / 8);
    }
}
