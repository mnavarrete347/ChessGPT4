import java.io.*;

public class UciTestClient {
    // How To Use UciTestClient
    // open your terminal in your IDE
    // make sure you have the java idk downloaded from Oracle
    // add the path to your system variables ex: "C:\Program Files\Java\jdk-21.0.10\bin"
    // type in the terminal javac src/Main.java
    // should see some .class files
    // then run the main

    public static void main(String[] args) {
        try {
            // Start your engine process
            Process process = new ProcessBuilder("java", "-cp", "bin;lib\\onnxruntime-1.17.0.jar", "Main").redirectErrorStream(true).start();

            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            // array so that the thead can modify it
            final long[] goStartTime = {0};
            // start a thread to continuously read engine output
            new Thread(() -> {
                String line;
                try {
                    while ((line = reader.readLine()) != null) {
                        System.out.println("ENGINE: " + line);

                        if (line.startsWith("bestmove")) {
                            long elapsed = System.currentTimeMillis() - goStartTime[0];
                            System.out.println("Response Time: " + elapsed + " ms");
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }).start();

            // send initializing commands
            send(writer, "uci", goStartTime);
            send(writer, "isready", goStartTime);
            send(writer, "ucinewgame", goStartTime);

            // set board using FEN
            String fen = buildFen(
                    "rnbqkbnr",
                    "pppppppp",
                    "8",
                    "8",
                    "8",   // white queen on e4
                    "8",
                    "PPPPPPPP",
                    "RNBQKBNR"
            );

            send(writer, "position fen " + fen, goStartTime);

            // Example: add moves
            // send(writer, "position startpos moves e2e4 e7e5", goStartTime);

            // Ask engine to think
            send(writer, "go movetime 10000", goStartTime);

            // Wait before quitting
            Thread.sleep(10000);

            send(writer, "quit", goStartTime);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static void send(BufferedWriter writer, String command, long[] goStartTime) throws IOException {
        System.out.println("CLIENT: " + command);

        if (command.startsWith("go")) {
            goStartTime[0] = System.currentTimeMillis();
        }

        writer.write(command);
        writer.newLine();
        writer.flush();
    }

    // takes in a variable length argument of Strings
    static String buildFen(String... ranks) {
        // ranks should be passed from rank 8 → rank 1
        return String.join("/", ranks) + " w KQkq - 0 1";
    }
}
