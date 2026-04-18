public final class Constants {

    private Constants() {}

    public static final boolean ENABLE_EN_PASSANT = false;

    public static final String MODEL_PATH    = "models/chess_model_EVH_150200.onnx";
    public static final String MOVE_MAP_PATH = "models/move_map_EVH_150200.ser";

    // Tapered evaluation phase weights
    public static final int KNIGHT_PHASE = 1;
    public static final int BISHOP_PHASE = 1;
    public static final int ROOK_PHASE   = 2;
    public static final int QUEEN_PHASE  = 4;
    public static final int MAX_PHASE    = 24;

    // Material values – midgame
    public static final int MG_PAWN   = 100;
    public static final int MG_KNIGHT = 320;
    public static final int MG_BISHOP = 330;
    public static final int MG_ROOK   = 500;
    public static final int MG_QUEEN  = 900;
    public static final int MG_KING   = 0;

    // Material values – endgame
    public static final int EG_PAWN   = 130;
    public static final int EG_KNIGHT = 320;
    public static final int EG_BISHOP = 350;
    public static final int EG_ROOK   = 500;
    public static final int EG_QUEEN  = 950;
    public static final int EG_KING   = 0;

    // Slider ray offsets
    public static final int[] ROOK_OFFSETS   = {8, -8, 1, -1};
    public static final int[] BISHOP_OFFSETS = {7, -7, 9, -9};
    public static final int[] QUEEN_OFFSETS  = {8, -8, 1, -1, 7, -7, 9, -9};

    // Bitboard file/rank masks
    public static final long FILE_A = 0x0101010101010101L;
    public static final long FILE_H = 0x8080808080808080L;
    public static final long RANK_1 = 0x00000000000000FFL;
    public static final long RANK_3 = 0x0000000000FF0000L;
    public static final long RANK_6 = 0x0000FF0000000000L;
    public static final long RANK_8 = 0xFF00000000000000L;

    // Precomputed knight and king attack tables
    public static final long[] KNIGHT_ATTACKS = new long[64];
    public static final long[] KING_ATTACKS   = new long[64];

    static {
        int[] knightDR = {-2, -2, -1, -1, 1, 1, 2, 2};
        int[] knightDF = {-1,  1, -2,  2,-2, 2,-1, 1};

        for (int sq = 0; sq < 64; sq++) {
            int r = sq / 8, f = sq % 8;
            long kn = 0L, ki = 0L;

            for (int d = 0; d < 8; d++) {
                int nr = r + knightDR[d], nf = f + knightDF[d];
                if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) kn |= 1L << (nr * 8 + nf);
            }
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) continue;
                    int nr = r + dr, nf = f + df;
                    if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) ki |= 1L << (nr * 8 + nf);
                }
            }
            KNIGHT_ATTACKS[sq] = kn;
            KING_ATTACKS[sq]   = ki;
        }
    }

    // Midgame PSTs (white's perspective; black mirrors with sq ^ 56)
    public static final int[] PAWN_PST = {
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    public static final int[] KNIGHT_PST = {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    };
    public static final int[] BISHOP_PST = {
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    };
    public static final int[] ROOK_PST = {
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  5, 10, 10,  5,  0,  0
    };
    public static final int[] QUEEN_PST = {
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
          0,  0,  5,  5,  5,  5,  0, -5,
         -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    };
    public static final int[] KING_PST = {
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30
    };

    // Endgame PSTs
    public static final int[] PAWN_PST_EG = {
         0,  0,  0,  0,  0,  0,  0,  0,
        -5, -5, -5, -5, -5, -5, -5, -5,
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 20, 20, 10, 10,  5,
        10, 20, 30, 40, 40, 30, 20, 10,
        30, 40, 50, 60, 60, 50, 40, 30,
        80, 80, 80, 80, 80, 80, 80, 80,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    public static final int[] KNIGHT_PST_EG = {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    };
    public static final int[] BISHOP_PST_EG = {
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    };
    public static final int[] ROOK_PST_EG = {
         0,  0,  0,  5,  5,  0,  0,  0,
         0,  0,  0, 10, 10,  0,  0,  0,
         0,  0,  0, 10, 10,  0,  0,  0,
         0,  0,  0, 10, 10,  0,  0,  0,
         0,  0,  0, 10, 10,  0,  0,  0,
         0,  0,  0, 10, 10,  0,  0,  0,
        20, 20, 20, 20, 20, 20, 20, 20,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    public static final int[] QUEEN_PST_EG = {
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5, 10, 10,  5,  0, -5,
         -5,  0,  5, 10, 10,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    };
    public static final int[] KING_PST_EG = {
        -50,-30,-30,-30,-30,-30,-30,-50,
        -30,-10,  0,  0,  0,  0,-10,-30,
        -30,  0, 20, 30, 30, 20,  0,-30,
        -30,  0, 30, 40, 40, 30,  0,-30,
        -30,  0, 30, 40, 40, 30,  0,-30,
        -30,  0, 20, 30, 30, 20,  0,-30,
        -30,-10,  0,  0,  0,  0,-10,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    };
}
