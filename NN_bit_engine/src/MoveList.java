public final class MoveList {

    public final int[] moves = new int[256];
    public int count = 0;

    public void add(int move) { moves[count++] = move; }
}
