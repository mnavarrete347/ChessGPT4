import ai.onnxruntime.*;
import java.util.*;

public class NeuralEngine {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    private final String[] indexToMoveUci;

    public NeuralEngine(String modelPath, Map<String, Integer> moveMap) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();

        int halfCores = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(halfCores);
        opts.setInterOpNumThreads(halfCores);

        this.session = env.createSession(modelPath, opts);
        this.inputName = session.getInputNames().iterator().next();

        int maxIdx = moveMap.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        this.indexToMoveUci = new String[maxIdx + 1];
        for (Map.Entry<String, Integer> e : moveMap.entrySet()) {
            indexToMoveUci[e.getValue()] = e.getKey();
        }
    }

    private OrtSession.Result runInference(Position pos, MoveList legal) throws OrtException {
        float[][][][] input = new float[1][13][8][8];

        long[] bbs = {
                pos.wp, pos.wn, pos.wb, pos.wr, pos.wq, pos.wk,
                pos.bp, pos.bn, pos.bb, pos.br, pos.bq, pos.bk
        };

        for (int ch = 0; ch < 12; ch++) {
            long bb = bbs[ch];
            while (bb != 0) {
                int sq = Long.numberOfTrailingZeros(bb);
                input[0][ch][sq / 8][sq % 8] = 1.0f;
                bb &= bb - 1;
            }
        }

        for (int i = 0; i < legal.count; i++) {
            int to = Move.getTo(legal.moves[i]);
            input[0][12][to / 8][to % 8] = 1.0f;
        }

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, input)) {
            return session.run(Collections.singletonMap(inputName, tensor));
        }
    }

    public double evaluatePosition(Position pos) throws OrtException {
        try (OrtSession.Result result = runInference(pos, pos.legalMoves())) {
            return ((float[][]) result.get(1).getValue())[0][0];
        }
    }

    public int topPolicyMove(Position pos, MoveList legal) throws OrtException {
        int[] top = topPolicyMoves(pos, legal, 1);
        return top.length == 0 ? 0 : top[0];
    }

    public int[] topPolicyMoves(Position pos, MoveList legal, int k) throws OrtException {
        if (legal == null || legal.count == 0 || k <= 0) return new int[0];

        try (OrtSession.Result result = runInference(pos, legal)) {
            float[] logits = ((float[][]) result.get(0).getValue())[0];

            int limit = Math.min(k, legal.count);
            int[] bestMoves = new int[limit];
            float[] bestScores = new float[limit];
            int found = 0;

            // Walk all logits once and keep only the best legal moves found so far.
            for (int i = 0; i < logits.length; i++) {
                if (i >= indexToMoveUci.length) continue;

                String uci = indexToMoveUci[i];
                if (uci == null) continue;

                int matchedMove = 0;
                for (int j = 0; j < legal.count; j++) {
                    int move = legal.moves[j];
                    if (Move.toUci(move).equals(uci)) {
                        matchedMove = move;
                        break;
                    }
                }

                if (matchedMove == 0) continue;

                // Skip duplicates
                boolean duplicate = false;
                for (int t = 0; t < found; t++) {
                    if (bestMoves[t] == matchedMove) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) continue;

                float score = logits[i];

                // Insert into sorted top-K arrays
                if (found < limit) {
                    int insert = found;
                    while (insert > 0 && score > bestScores[insert - 1]) {
                        bestScores[insert] = bestScores[insert - 1];
                        bestMoves[insert] = bestMoves[insert - 1];
                        insert--;
                    }
                    bestScores[insert] = score;
                    bestMoves[insert] = matchedMove;
                    found++;
                } else if (score > bestScores[limit - 1]) {
                    int insert = limit - 1;
                    while (insert > 0 && score > bestScores[insert - 1]) {
                        bestScores[insert] = bestScores[insert - 1];
                        bestMoves[insert] = bestMoves[insert - 1];
                        insert--;
                    }
                    bestScores[insert] = score;
                    bestMoves[insert] = matchedMove;
                }
            }

            return found == limit ? bestMoves : java.util.Arrays.copyOf(bestMoves, found);
        }
    }
}