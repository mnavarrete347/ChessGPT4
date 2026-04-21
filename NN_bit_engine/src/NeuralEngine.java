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

    public int topPolicyMove(Position pos, MoveList legal) throws OrtException {
        int[] top = topPolicyMoves(pos, legal, 1);
        return top.length == 0 ? 0 : top[0];
    }

    public int[] topPolicyMoves(Position pos, MoveList legal, int k) throws OrtException {
        if (legal == null || legal.count == 0 || k <= 0) return new int[0];

        try (OrtSession.Result result = runInference(pos, legal)) {
            float[] logits = ((float[][]) result.get(0).getValue())[0];

            Integer[] idx = new Integer[logits.length];
            for (int i = 0; i < logits.length; i++) idx[i] = i;
            Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));

            int[] tmp = new int[Math.min(k, legal.count)];
            int found = 0;

            for (int i : idx) {
                if (i >= indexToMoveUci.length || indexToMoveUci[i] == null) continue;
                String uci = indexToMoveUci[i];

                for (int j = 0; j < legal.count; j++) {
                    int move = legal.moves[j];
                    if (Move.toUci(move).equals(uci)) {
                        boolean duplicate = false;
                        for (int t = 0; t < found; t++) {
                            if (tmp[t] == move) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            tmp[found++] = move;
                            if (found == tmp.length) {
                                return Arrays.copyOf(tmp, found);
                            }
                        }
                        break;
                    }
                }
            }
            return Arrays.copyOf(tmp, found);
        }
    }

    // Returns a map from legal move -> rank score (higher is better), for top-K moves only.
    // Example scores:
    //   best gets 10000, next 9000, next 8000, ...
    public Map<Integer, Integer> policyScores(Position pos, MoveList legal, int k) throws OrtException {
        Map<Integer, Integer> out = new HashMap<>();
        int[] top = topPolicyMoves(pos, legal, k);
        for (int i = 0; i < top.length; i++) {
            out.put(top[i], 10000 - (i * 1000));
        }
        return out;
    }
}