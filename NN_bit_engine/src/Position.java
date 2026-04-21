public final class Position {

    // Piece bitboards
    public final long wp, wn, wb, wr, wq, wk;
    public final long bp, bn, bb, br, bq, bk;

    // Aggregate occupancy
    public final long whitePieces, blackPieces, allPieces;

    // Game state
    public final boolean whiteToMove;
    public final boolean whiteKingSideCastle, whiteQueenSideCastle;
    public final boolean blackKingSideCastle, blackQueenSideCastle;

    // Incremental evaluation (white-relative)
    public int score;

    // Cached cheap-search helpers
    public int material;
    public final long key;

    // Z-flags for incremental key update
    private static final long[][] Z_PIECE = new long[12][64];
    private static final long[] Z_CASTLING = new long[4]; // WK, WQ, BK, BQ
    private static final long Z_SIDE;

    static {
        java.util.Random rng = new java.util.Random(4318L);

        for (int p = 0; p < 12; p++) {
            for (int sq = 0; sq < 64; sq++) {
                Z_PIECE[p][sq] = rng.nextLong();
            }
        }
        for (int i = 0; i < 4; i++) {
            Z_CASTLING[i] = rng.nextLong();
        }
        Z_SIDE = rng.nextLong();
    }

    public Position(long[] pieces, boolean whiteToMove,
                    boolean wK, boolean wQ, boolean bK, boolean bQ) {
        this.wp = pieces[0]; this.wn = pieces[1]; this.wb = pieces[2];
        this.wr = pieces[3]; this.wq = pieces[4]; this.wk = pieces[5];
        this.bp = pieces[6]; this.bn = pieces[7]; this.bb = pieces[8];
        this.br = pieces[9]; this.bq = pieces[10]; this.bk = pieces[11];

        this.whitePieces = wp | wn | wb | wr | wq | wk;
        this.blackPieces = bp | bn | bb | br | bq | bk;
        this.allPieces   = whitePieces | blackPieces;

        this.whiteToMove          = whiteToMove;
        this.whiteKingSideCastle  = wK;
        this.whiteQueenSideCastle = wQ;
        this.blackKingSideCastle  = bK;
        this.blackQueenSideCastle = bQ;

        this.material = computeMaterial();
        this.key      = computeKey();
    }

    private Position(long[] pieces, boolean whiteToMove,
                     boolean wK, boolean wQ, boolean bK, boolean bQ,
                     int material, long key) {
        this.wp = pieces[0]; this.wn = pieces[1]; this.wb = pieces[2];
        this.wr = pieces[3]; this.wq = pieces[4]; this.wk = pieces[5];
        this.bp = pieces[6]; this.bn = pieces[7]; this.bb = pieces[8];
        this.br = pieces[9]; this.bq = pieces[10]; this.bk = pieces[11];

        this.whitePieces = wp | wn | wb | wr | wq | wk;
        this.blackPieces = bp | bn | bb | br | bq | bk;
        this.allPieces   = whitePieces | blackPieces;

        this.whiteToMove          = whiteToMove;
        this.whiteKingSideCastle  = wK;
        this.whiteQueenSideCastle = wQ;
        this.blackKingSideCastle  = bK;
        this.blackQueenSideCastle = bQ;

        this.material = material;
        this.key      = key;
    }

    public static Position startPos() {
        return fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    public static Position fromFEN(String fen) {
        String[] parts = fen.trim().split("\\s+");
        boolean  wtm   = parts.length <= 1 || parts[1].equals("w");
        long[]   p     = new long[12];
        int      rank  = 7, file = 0;

        for (char c : parts[0].toCharArray()) {
            if (c == '/') { rank--; file = 0; }
            else if (Character.isDigit(c)) { file += c - '0'; }
            else {
                int idx = pieceIndex(c);
                if (idx >= 0) p[idx] |= 1L << (rank * 8 + file);
                file++;
            }
        }

        boolean[] cr  = parseCastlingRights(parts.length > 2 ? parts[2] : "-");

        Position pos = new Position(p, wtm, cr[0], cr[1], cr[2], cr[3]);
        pos.calculateInitialScores();
        return pos;
    }

    // -------------------------------------------------------------------------
    // Incremental scoring
    // -------------------------------------------------------------------------

    private void calculateInitialScores() {
        score = 0;
        for (int i = 0; i < 12; i++) {
            long bb = bbByIndex(i);
            while (bb != 0) {
                int sq = Long.numberOfTrailingZeros(bb);
                score += value(i, sq);
                bb &= bb - 1;
            }
        }
    }

    static int value(int idx, int sq) {
        boolean w = idx < 6;
        int t = w ? sq : sq ^ 56;
        int s = switch (idx % 6) {
            case 0 -> Constants.PAWN   + Constants.PAWN_PST[t];
            case 1 -> Constants.KNIGHT + Constants.KNIGHT_PST[t];
            case 2 -> Constants.BISHOP + Constants.BISHOP_PST[t];
            case 3 -> Constants.ROOK   + Constants.ROOK_PST[t];
            case 4 -> Constants.QUEEN  + Constants.QUEEN_PST[t];
            default-> Constants.KING   + Constants.KING_PST[t];
        };
        return w ? s : -s;
    }

    // -------------------------------------------------------------------------
    // Make move
    // -------------------------------------------------------------------------

    public Position makeMove(int move) {
        long[] next = {wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk};
        int from  = Move.getFrom(move);
        int to    = Move.getTo(move);
        int promo = Move.getPromo(move);

        long fBit = 1L << from;
        long tBit = 1L << to;

        int  nScore    = score;
        int  nMaterial = material;
        long nKey      = key;

        // Save old castling rights so we can update the hash if they change
        boolean oldWK = whiteKingSideCastle, oldWQ = whiteQueenSideCastle;
        boolean oldBK = blackKingSideCastle, oldBQ = blackQueenSideCastle;

        // Identify moving piece
        int movIdx = -1;
        for (int i = 0; i < 12; i++) {
            if ((next[i] & fBit) != 0) {
                movIdx = i;
                break;
            }
        }
        if (movIdx == -1) {
            throw new IllegalStateException("No moving piece found for move: " + move);
        }

        // Remove moving piece from source square
        nScore -= value(movIdx, from);
        nKey   ^= Z_PIECE[movIdx][from];

        // Capture
        if ((allPieces & tBit) != 0) {
            for (int i = 0; i < 12; i++) {
                if ((next[i] & tBit) != 0) {
                    nScore -= value(i, to);
                    nKey   ^= Z_PIECE[i][to];

                    switch (i % 6) {
                        case 0 -> nMaterial -= Constants.PAWN;
                        case 1 -> nMaterial -= Constants.KNIGHT;
                        case 2 -> nMaterial -= Constants.BISHOP;
                        case 3 -> nMaterial -= Constants.ROOK;
                        case 4 -> nMaterial -= Constants.QUEEN;
                        default -> { /* king not counted in material */ }
                    }

                    next[i] ^= tBit;
                    break;
                }
            }
        }

        // Move piece
        next[movIdx] ^= fBit | tBit;
        nScore += value(movIdx, to);
        nKey   ^= Z_PIECE[movIdx][to];

        // Promotion
        if (promo != 0 && (movIdx == 0 || movIdx == 6)) {
            // Remove pawn from promoted square
            nScore -= value(movIdx, to);
            nKey   ^= Z_PIECE[movIdx][to];
            next[movIdx] ^= tBit;

            int pIdx = whiteToMove
                    ? (promo == 1 ? 4 : promo == 2 ? 3 : promo == 3 ? 2 : 1)
                    : (promo == 1 ? 10 : promo == 2 ? 9 : promo == 3 ? 8 : 7);

            next[pIdx] |= tBit;
            nScore += value(pIdx, to);
            nKey   ^= Z_PIECE[pIdx][to];

            nMaterial -= Constants.PAWN;
            nMaterial += switch (promo) {
                case 1 -> Constants.QUEEN;
                case 2 -> Constants.ROOK;
                case 3 -> Constants.BISHOP;
                default -> Constants.KNIGHT;
            };
        }

        // Update castling rights
        boolean wK = whiteKingSideCastle, wQ = whiteQueenSideCastle;
        boolean bK = blackKingSideCastle, bQ = blackQueenSideCastle;

        if (from == 4  || to == 4)  { wK = false; wQ = false; }
        if (from == 60 || to == 60) { bK = false; bQ = false; }
        if (from == 0  || to == 0)  wQ = false;
        if (from == 7  || to == 7)  wK = false;
        if (from == 56 || to == 56) bQ = false;
        if (from == 63 || to == 63) bK = false;

        // Update castling-right hash
        if (oldWK != wK) nKey ^= Z_CASTLING[0];
        if (oldWQ != wQ) nKey ^= Z_CASTLING[1];
        if (oldBK != bK) nKey ^= Z_CASTLING[2];
        if (oldBQ != bQ) nKey ^= Z_CASTLING[3];

        // Castling rook moves
        if (movIdx == 5 && from == 4) {
            if (to == 6) {
                nScore += value(3, 5) - value(3, 7);
                next[3] ^= (1L << 7) | (1L << 5);
                nKey   ^= Z_PIECE[3][7];
                nKey   ^= Z_PIECE[3][5];
            } else if (to == 2) {
                nScore += value(3, 3) - value(3, 0);
                next[3] ^= (1L << 0) | (1L << 3);
                nKey   ^= Z_PIECE[3][0];
                nKey   ^= Z_PIECE[3][3];
            }
        } else if (movIdx == 11 && from == 60) {
            if (to == 62) {
                nScore += value(9, 61) - value(9, 63);
                next[9] ^= (1L << 63) | (1L << 61);
                nKey   ^= Z_PIECE[9][63];
                nKey   ^= Z_PIECE[9][61];
            } else if (to == 58) {
                nScore += value(9, 59) - value(9, 56);
                next[9] ^= (1L << 56) | (1L << 59);
                nKey   ^= Z_PIECE[9][56];
                nKey   ^= Z_PIECE[9][59];
            }
        }

        // Flip side to move
        nKey ^= Z_SIDE;

        Position result = new Position(next, !whiteToMove, wK, wQ, bK, bQ, nMaterial, nKey);
        result.score = nScore;
        return result;
    }

    // -------------------------------------------------------------------------
    // Move generation
    // -------------------------------------------------------------------------

    public MoveList pseudoLegalMoves(MoveList list) {
        list.clear();
        boolean w = whiteToMove;
        genPawn(list, w);
        genKnight(list, w);
        genSliders(list, w, Constants.BISHOP_OFFSETS, w ? wb : bb);
        genSliders(list, w, Constants.ROOK_OFFSETS,   w ? wr : br);
        genSliders(list, w, Constants.QUEEN_OFFSETS,  w ? wq : bq);
        genKing(list, w);
        return list;
    }

    public MoveList legalMoves(MoveList pseudo, MoveList legal) {
        pseudoLegalMoves(pseudo);
        legal.clear();
        for (int i = 0; i < pseudo.count; i++) {
            Position next = makeMove(pseudo.moves[i]);
            if (!next.inCheck(!next.whiteToMove)) legal.add(pseudo.moves[i]);
        }
        return legal;
    }

    private void genPawn(MoveList list, boolean w) {
        long enemy = w ? blackPieces : whitePieces;
        long empty = ~allPieces;

        if (w) {
            // move up by shifting left by 8
            long single = (wp << 8) & empty;
            addPawnMoves(list, single & ~Constants.RANK_8, 8, false);
            addPawnMoves(list, single &  Constants.RANK_8, 8, true);
            addPawnMoves(list, ((single & Constants.RANK_3) << 8) & empty, 16, false);
            long cL = (wp << 7) & enemy & ~Constants.FILE_H;
            addPawnMoves(list, cL & ~Constants.RANK_8, 7, false);
            addPawnMoves(list, cL &  Constants.RANK_8, 7, true);
            long cR = (wp << 9) & enemy & ~Constants.FILE_A;
            addPawnMoves(list, cR & ~Constants.RANK_8, 9, false);
            addPawnMoves(list, cR &  Constants.RANK_8, 9, true);
        } else {
            // move down by shifting right by 8
            long single = (bp >> 8) & empty;
            addPawnMoves(list, single & ~Constants.RANK_1, -8, false);
            addPawnMoves(list, single &  Constants.RANK_1, -8, true);
            addPawnMoves(list, ((single & Constants.RANK_6) >> 8) & empty, -16, false);
            long cL = (bp >> 9) & enemy & ~Constants.FILE_H;
            addPawnMoves(list, cL & ~Constants.RANK_1, -9, false);
            addPawnMoves(list, cL &  Constants.RANK_1, -9, true);
            long cR = (bp >> 7) & enemy & ~Constants.FILE_A;
            addPawnMoves(list, cR & ~Constants.RANK_1, -7, false);
            addPawnMoves(list, cR &  Constants.RANK_1, -7, true);
        }
    }

    private void genKnight(MoveList list, boolean w) {
        long pieces   = w ? wn : bn;
        long friendly = w ? whitePieces : blackPieces;
        while (pieces != 0) {
            int from = Long.numberOfTrailingZeros(pieces);
            serializeMoves(list, from, Constants.KNIGHT_ATTACKS[from] & ~friendly);
            pieces &= pieces - 1;
        }
    }

    private void genKing(MoveList list, boolean w) {
        long kingBoard = w ? wk : bk;
        if (kingBoard == 0) return;

        int  from     = Long.numberOfTrailingZeros(kingBoard);
        long friendly = w ? whitePieces : blackPieces;

        serializeMoves(list, from, Constants.KING_ATTACKS[from] & ~friendly);
        if (w) {
            if (whiteKingSideCastle  && (allPieces & 0x60L) == 0
                    && !isSquareAttacked(4, false)
                    && !isSquareAttacked(5, false)
                    && !isSquareAttacked(6, false))
                list.add(Move.create(4, 6, 0));
            if (whiteQueenSideCastle && (allPieces & 0x0EL) == 0
                    && !isSquareAttacked(4, false)
                    && !isSquareAttacked(3, false)
                    && !isSquareAttacked(2, false))
                list.add(Move.create(4, 2, 0));
        } else {
            if (blackKingSideCastle  && (allPieces & 0x6000000000000000L) == 0
                    && !isSquareAttacked(60, true)
                    && !isSquareAttacked(61, true)
                    && !isSquareAttacked(62, true))
                list.add(Move.create(60, 62, 0));
            if (blackQueenSideCastle && (allPieces & 0x0E00000000000000L) == 0
                    && !isSquareAttacked(60, true)
                    && !isSquareAttacked(59, true)
                    && !isSquareAttacked(58, true))
                list.add(Move.create(60, 58, 0));
        }
    }

    private void genSliders(MoveList list, boolean w, int[] offsets, long pieces) {
        long friendly = w ? whitePieces : blackPieces;
        long enemy    = w ? blackPieces : whitePieces;
        while (pieces != 0) {
            int from = Long.numberOfTrailingZeros(pieces);
            for (int offset : offsets) {
                int cur = from;
                while (true) {
                    int nxt = cur + offset;
                    if (nxt < 0 || nxt >= 64 || isWrap(cur, nxt, offset)) break;
                    long bit = 1L << nxt;
                    if ((friendly & bit) != 0) break;
                    list.add(Move.create(from, nxt, 0));
                    if ((enemy & bit) != 0) break;
                    cur = nxt;
                }
            }
            pieces &= pieces - 1;
        }
    }

    private void serializeMoves(MoveList list, int from, long dests) {
        while (dests != 0) {
            list.add(Move.create(from, Long.numberOfTrailingZeros(dests), 0));
            dests &= dests - 1;
        }
    }

    private void addPawnMoves(MoveList list, long dests, int offset, boolean promo) {
        while (dests != 0) {
            int to = Long.numberOfTrailingZeros(dests), from = to - offset;
            if (promo) {
                list.add(Move.create(from, to, 1));
                list.add(Move.create(from, to, 4));
                list.add(Move.create(from, to, 2));
                list.add(Move.create(from, to, 3));
            } else {
                list.add(Move.create(from, to, 0));
            }
            dests &= dests - 1;
        }
    }

    // -------------------------------------------------------------------------
    // Attack detection
    // -------------------------------------------------------------------------

    public boolean inCheck(boolean isWhiteKing) {
        long king = isWhiteKing ? wk : bk;
        if (king == 0) return true;
        return isSquareAttacked(Long.numberOfTrailingZeros(king), !isWhiteKing);
    }

    public boolean isSquareAttacked(int sq, boolean byWhite) {
        long bit = 1L << sq;

        if (byWhite) {
            if (((bit >> 7) & wp & ~Constants.FILE_A) != 0) return true;
            if (((bit >> 9) & wp & ~Constants.FILE_H) != 0) return true;
        } else {
            if (((bit << 7) & bp & ~Constants.FILE_H) != 0) return true;
            if (((bit << 9) & bp & ~Constants.FILE_A) != 0) return true;
        }

        if ((Constants.KNIGHT_ATTACKS[sq] & (byWhite ? wn : bn)) != 0) return true;
        if ((Constants.KING_ATTACKS[sq]   & (byWhite ? wk : bk)) != 0) return true;

        long straight = byWhite ? (wr | wq) : (br | bq);
        long diagonal = byWhite ? (wb | wq) : (bb | bq);

        for (int offset : Constants.ROOK_OFFSETS) {
            int cur = sq;
            while (true) {
                int nxt = cur + offset;
                if (nxt < 0 || nxt >= 64 || isWrap(cur, nxt, offset)) break;
                long b = 1L << nxt;
                if ((straight & b) != 0) return true;
                if ((allPieces & b) != 0) break;
                cur = nxt;
            }
        }
        for (int offset : Constants.BISHOP_OFFSETS) {
            int cur = sq;
            while (true) {
                int nxt = cur + offset;
                if (nxt < 0 || nxt >= 64 || isWrap(cur, nxt, offset)) break;
                long b = 1L << nxt;
                if ((diagonal & b) != 0) return true;
                if ((allPieces & b) != 0) break;
                cur = nxt;
            }
        }
        return false;
    }

    private boolean isWrap(int from, int to, int offset) {
        int diff = Math.abs((to & 7) - (from & 7));
        return Math.abs(offset) == 8 ? diff != 0 : diff != 1;
    }

    // -------------------------------------------------------------------------
    // Display and helpers
    // -------------------------------------------------------------------------

    public char getPieceAt(int sq) {
        long b = 1L << sq;
        if ((wp & b) != 0) return 'P'; if ((wn & b) != 0) return 'N';
        if ((wb & b) != 0) return 'B'; if ((wr & b) != 0) return 'R';
        if ((wq & b) != 0) return 'Q'; if ((wk & b) != 0) return 'K';
        if ((bp & b) != 0) return 'p'; if ((bn & b) != 0) return 'n';
        if ((bb & b) != 0) return 'b'; if ((br & b) != 0) return 'r';
        if ((bq & b) != 0) return 'q'; if ((bk & b) != 0) return 'k';
        return '.';
    }

    private static int pieceIndex(char c) {
        return switch (c) {
            case 'P' -> 0; case 'N' -> 1; case 'B' -> 2;
            case 'R' -> 3; case 'Q' -> 4; case 'K' -> 5;
            case 'p' -> 6; case 'n' -> 7; case 'b' -> 8;
            case 'r' -> 9; case 'q' -> 10; case 'k' -> 11;
            default  -> -1;
        };
    }

    private static boolean[] parseCastlingRights(String s) {
        return new boolean[]{s.contains("K"), s.contains("Q"), s.contains("k"), s.contains("q")};
    }

    private long bbByIndex(int i) {
        return switch (i) {
            case  0 -> wp; case  1 -> wn; case  2 -> wb;
            case  3 -> wr; case  4 -> wq; case  5 -> wk;
            case  6 -> bp; case  7 -> bn; case  8 -> bb;
            case  9 -> br; case 10 -> bq; case 11 -> bk;
            default -> 0L;
        };
    }

    private int computeMaterial() {
        return Long.bitCount(wp | bp) * Constants.PAWN +
                Long.bitCount(wn | bn) * Constants.KNIGHT +
                Long.bitCount(wb | bb) * Constants.BISHOP +
                Long.bitCount(wr | br) * Constants.ROOK +
                Long.bitCount(wq | bq) * Constants.QUEEN;
    }

    public int totalMaterial() {
        return material;
    }

    private long computeKey() {
        long h = 0L;

        for (int i = 0; i < 12; i++) {
            long bb = bbByIndex(i);
            while (bb != 0) {
                int sq = Long.numberOfTrailingZeros(bb);
                h ^= Z_PIECE[i][sq];
                bb &= bb - 1;
            }
        }

        if (!whiteToMove) h ^= Z_SIDE;
        if (whiteKingSideCastle)  h ^= Z_CASTLING[0];
        if (whiteQueenSideCastle) h ^= Z_CASTLING[1];
        if (blackKingSideCastle)  h ^= Z_CASTLING[2];
        if (blackQueenSideCastle) h ^= Z_CASTLING[3];

        return h;
    }
}