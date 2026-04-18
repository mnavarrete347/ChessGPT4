import ai.onnxruntime.*;
import java.util.*;

public class NeuralEngine {

    private final OrtEnvironment env;
    private final OrtSession     session;
    private final String         inputName;
    private final String[]       indexToMoveUci;

    public NeuralEngine(String modelPath, Map<String, Integer> moveMap) throws OrtException {
        this.env       = OrtEnvironment.getEnvironment();
        this.session   = env.createSession(modelPath, new OrtSession.SessionOptions());
        this.inputName = session.getInputNames().iterator().next();

        int maxIdx = moveMap.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        this.indexToMoveUci = new String[maxIdx + 1];
        for (Map.Entry<String, Integer> e : moveMap.entrySet())
            indexToMoveUci[e.getValue()] = e.getKey();
    }

    // Builds [1,13,8,8] input tensor, runs both output heads, returns raw result.
    // Caller must close the result.
    private OrtSession.Result runInference(Position pos, MoveList legal) throws OrtException {
        float[][][][] input = new float[1][13][8][8];

        long[] bbs = {pos.wp,pos.wn,pos.wb,pos.wr,pos.wq,pos.wk,
                      pos.bp,pos.bn,pos.bb,pos.br,pos.bq,pos.bk};
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

    // Returns value-head scalar in (-1,+1), current side's perspective.
    public double evaluatePosition(Position pos) throws OrtException {
        try (OrtSession.Result result = runInference(pos, pos.legalMoves())) {
            return ((float[][]) result.get(1).getValue())[0][0];
        }
    }

    // Returns highest-logit legal move from the policy head; 0 if none match.
    public int topPolicyMove(Position pos, MoveList legal) throws OrtException {
        try (OrtSession.Result result = runInference(pos, legal)) {
            float[] logits = ((float[][]) result.get(0).getValue())[0];

            Integer[] idx = new Integer[logits.length];
            for (int i = 0; i < logits.length; i++) idx[i] = i;
            Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));

            for (int i : idx) {
                if (i >= indexToMoveUci.length || indexToMoveUci[i] == null) continue;
                String uci = indexToMoveUci[i];
                for (int j = 0; j < legal.count; j++)
                    if (Move.toUci(legal.moves[j]).equals(uci)) return legal.moves[j];
            }
            return 0;
        }
    }
}
