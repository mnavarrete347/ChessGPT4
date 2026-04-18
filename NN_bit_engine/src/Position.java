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
    public final int enPassantSq;

    // Incremental evaluation (white-relative)
    public int mgScore, egScore, phase;

    public Position(long[] pieces, boolean whiteToMove,
                    boolean wK, boolean wQ, boolean bK, boolean bQ, int epSq) {
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
        this.enPassantSq          = epSq;
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
        String    ept = parts.length > 3 ? parts[3] : "-";
        int       epSq = ept.equals("-") ? -1 : Move.squareIndex(ept);

        Position pos = new Position(p, wtm, cr[0], cr[1], cr[2], cr[3], epSq);
        pos.calculateInitialScores();
        return pos;
    }

    // -------------------------------------------------------------------------
    // Incremental scoring
    // -------------------------------------------------------------------------

    private void calculateInitialScores() {
        mgScore = 0; egScore = 0; phase = 0;
        for (int i = 0; i < 12; i++) {
            long bb = bbByIndex(i);
            while (bb != 0) {
                int sq = Long.numberOfTrailingZeros(bb);
                mgScore += mgValue(i, sq);
                egScore += egValue(i, sq);
                phase   += piecePhase(i);
                bb &= bb - 1;
            }
        }
    }

    static int mgValue(int idx, int sq) {
        boolean w = idx < 6;
        int t = w ? sq : sq ^ 56;
        int s = switch (idx % 6) {
            case 0 -> Constants.MG_PAWN   + Constants.PAWN_PST[t];
            case 1 -> Constants.MG_KNIGHT + Constants.KNIGHT_PST[t];
            case 2 -> Constants.MG_BISHOP + Constants.BISHOP_PST[t];
            case 3 -> Constants.MG_ROOK   + Constants.ROOK_PST[t];
            case 4 -> Constants.MG_QUEEN  + Constants.QUEEN_PST[t];
            default-> Constants.MG_KING   + Constants.KING_PST[t];
        };
        return w ? s : -s;
    }

    static int egValue(int idx, int sq) {
        boolean w = idx < 6;
        int t = w ? sq : sq ^ 56;
        int s = switch (idx % 6) {
            case 0 -> Constants.EG_PAWN   + Constants.PAWN_PST_EG[t];
            case 1 -> Constants.EG_KNIGHT + Constants.KNIGHT_PST_EG[t];
            case 2 -> Constants.EG_BISHOP + Constants.BISHOP_PST_EG[t];
            case 3 -> Constants.EG_ROOK   + Constants.ROOK_PST_EG[t];
            case 4 -> Constants.EG_QUEEN  + Constants.QUEEN_PST_EG[t];
            default-> Constants.EG_KING   + Constants.KING_PST_EG[t];
        };
        return w ? s : -s;
    }

    static int piecePhase(int idx) {
        return switch (idx % 6) {
            case 1 -> Constants.KNIGHT_PHASE;
            case 2 -> Constants.BISHOP_PHASE;
            case 3 -> Constants.ROOK_PHASE;
            case 4 -> Constants.QUEEN_PHASE;
            default -> 0;
        };
    }

    // -------------------------------------------------------------------------
    // Make move
    // -------------------------------------------------------------------------

    public Position makeMove(int move) {
        long[] next = {wp,wn,wb,wr,wq,wk,bp,bn,bb,br,bq,bk};
        int from  = Move.getFrom(move), to = Move.getTo(move), promo = Move.getPromo(move);
        long fBit = 1L << from, tBit = 1L << to;
        int nMg = mgScore, nEg = egScore, nPh = phase;

        // Identify moving piece
        int movIdx = -1;
        for (int i = 0; i < 12; i++) if ((next[i] & fBit) != 0) { movIdx = i; break; }

        nMg -= mgValue(movIdx, from);
        nEg -= egValue(movIdx, from);

        // Capture
        if ((allPieces & tBit) != 0) {
            for (int i = 0; i < 12; i++) {
                if ((next[i] & tBit) != 0) {
                    nMg -= mgValue(i, to); nEg -= egValue(i, to); nPh -= piecePhase(i);
                    next[i] ^= tBit;
                    break;
                }
            }
        }

        // En passant capture
        int nextEpSq = -1;
        if (Constants.ENABLE_EN_PASSANT && (movIdx == 0 || movIdx == 6) && to == enPassantSq) {
            int capSq  = whiteToMove ? to - 8 : to + 8;
            int vicIdx = whiteToMove ? 6 : 0;
            nMg -= mgValue(vicIdx, capSq); nEg -= egValue(vicIdx, capSq);
            next[vicIdx] ^= 1L << capSq;
        }

        // Move piece
        next[movIdx] ^= fBit | tBit;
        nMg += mgValue(movIdx, to);
        nEg += egValue(movIdx, to);

        // Promotion
        if (promo != 0 && (movIdx == 0 || movIdx == 6)) {
            nMg -= mgValue(movIdx, to); nEg -= egValue(movIdx, to);
            next[movIdx] ^= tBit;
            int pIdx = whiteToMove
                    ? (promo == 1 ? 4 : promo == 2 ? 3 : promo == 3 ? 2 : 1)
                    : (promo == 1 ? 10 : promo == 2 ? 9 : promo == 3 ? 8 : 7);
            next[pIdx] |= tBit;
            nMg += mgValue(pIdx, to); nEg += egValue(pIdx, to); nPh += piecePhase(pIdx);
        }

        // Double pawn push → set EP square
        if (Constants.ENABLE_EN_PASSANT && (movIdx == 0 || movIdx == 6) && Math.abs(to - from) == 16)
            nextEpSq = (from + to) / 2;

        // Update castling rights
        // Bug fix: original used movIdx==5 instead of checking from==4 for the king square,
        // which incorrectly revoked rights when a non-king piece on the king square moved.
        boolean wK = whiteKingSideCastle, wQ = whiteQueenSideCastle;
        boolean bK = blackKingSideCastle, bQ = blackQueenSideCastle;
        if (from == 4  || to == 4)  { wK = false; wQ = false; }
        if (from == 60 || to == 60) { bK = false; bQ = false; }
        if (from == 0  || to == 0)  wQ = false;
        if (from == 7  || to == 7)  wK = false;
        if (from == 56 || to == 56) bQ = false;
        if (from == 63 || to == 63) bK = false;

        // Castling rook moves
        if (movIdx == 5 && from == 4) {
            if (to == 6) {
                nMg += mgValue(3,5) - mgValue(3,7); nEg += egValue(3,5) - egValue(3,7);
                next[3] ^= (1L << 7) | (1L << 5);
            } else if (to == 2) {
                nMg += mgValue(3,3) - mgValue(3,0); nEg += egValue(3,3) - egValue(3,0);
                next[3] ^= 1L | (1L << 3);
            }
        } else if (movIdx == 11 && from == 60) {
            if (to == 62) {
                nMg += mgValue(9,61) - mgValue(9,63); nEg += egValue(9,61) - egValue(9,63);
                next[9] ^= (1L << 63) | (1L << 61);
            } else if (to == 58) {
                nMg += mgValue(9,59) - mgValue(9,56); nEg += egValue(9,59) - egValue(9,56);
                next[9] ^= (1L << 56) | (1L << 59);
            }
        }

        Position result = new Position(next, !whiteToMove, wK, wQ, bK, bQ, nextEpSq);
        result.mgScore  = nMg;
        result.egScore  = nEg;
        result.phase    = nPh;
        return result;
    }

    // -------------------------------------------------------------------------
    // Move generation
    // -------------------------------------------------------------------------

    public MoveList pseudoLegalMoves() {
        MoveList list = new MoveList();
        boolean  w    = whiteToMove;
        genPawn(list, w);
        genKnight(list, w);
        genSliders(list, w, Constants.BISHOP_OFFSETS, w ? wb : bb);
        genSliders(list, w, Constants.ROOK_OFFSETS,   w ? wr : br);
        genSliders(list, w, Constants.QUEEN_OFFSETS,  w ? wq : bq);
        genKing(list, w);
        return list;
    }

    public MoveList legalMoves() {
        MoveList pseudo = pseudoLegalMoves();
        MoveList legal  = new MoveList();
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
            if (Constants.ENABLE_EN_PASSANT && enPassantSq != -1) {
                long ep = 1L << enPassantSq;
                if (((wp << 7) & ep & ~Constants.FILE_H) != 0) list.add(Move.create(enPassantSq - 7, enPassantSq, 0));
                if (((wp << 9) & ep & ~Constants.FILE_A) != 0) list.add(Move.create(enPassantSq - 9, enPassantSq, 0));
            }
        } else {
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
            if (Constants.ENABLE_EN_PASSANT && enPassantSq != -1) {
                long ep = 1L << enPassantSq;
                if (((bp >> 9) & ep & ~Constants.FILE_H) != 0) list.add(Move.create(enPassantSq + 9, enPassantSq, 0));
                if (((bp >> 7) & ep & ~Constants.FILE_A) != 0) list.add(Move.create(enPassantSq + 7, enPassantSq, 0));
            }
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
        // If the king is missing, do not attempt to generate moves
        if (kingBoard == 0) return;

        int  from     = Long.numberOfTrailingZeros(kingBoard);
        long friendly = w ? whitePieces : blackPieces;

        serializeMoves(list, from, Constants.KING_ATTACKS[from] & ~friendly);
        //  Castling logic
        if (w) {
            if (whiteKingSideCastle  && (allPieces & 0x60L) == 0
                    && !isSquareAttacked(4, false) && !isSquareAttacked(5, false) && !isSquareAttacked(6, false))
                list.add(Move.create(4, 6, 0));
            if (whiteQueenSideCastle && (allPieces & 0x0EL) == 0
                    && !isSquareAttacked(4, false) && !isSquareAttacked(3, false) && !isSquareAttacked(2, false))
                list.add(Move.create(4, 2, 0));
        } else {
            if (blackKingSideCastle  && (allPieces & 0x6000000000000000L) == 0
                    && !isSquareAttacked(60, true) && !isSquareAttacked(61, true) && !isSquareAttacked(62, true))
                list.add(Move.create(60, 62, 0));
            if (blackQueenSideCastle && (allPieces & 0x0E00000000000000L) == 0
                    && !isSquareAttacked(60, true) && !isSquareAttacked(59, true) && !isSquareAttacked(58, true))
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
                if ((straight   & b) != 0) return true;
                if ((allPieces  & b) != 0) break;
                cur = nxt;
            }
        }
        for (int offset : Constants.BISHOP_OFFSETS) {
            int cur = sq;
            while (true) {
                int nxt = cur + offset;
                if (nxt < 0 || nxt >= 64 || isWrap(cur, nxt, offset)) break;
                long b = 1L << nxt;
                if ((diagonal  & b) != 0) return true;
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
}
