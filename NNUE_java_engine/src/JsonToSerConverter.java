import java.io.*;
import java.util.*;
import org.json.*;

public class JsonToSerConverter {

    public static void main(String[] args) {
        try {
            // Read JSON file
            String content = new String(
                    java.nio.file.Files.readAllBytes(
                            java.nio.file.Paths.get("models\\move_map.json")
                    )
            );

            JSONObject json = new JSONObject(content);

            Map<String, Integer> map = new HashMap<>();

            for (String key : json.keySet()) {
                map.put(key, json.getInt(key));
            }

            // Save as .ser file
            ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream("models\\move_map.ser")
            );

            oos.writeObject(map);
            oos.close();

            System.out.println("Saved move_map.ser");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
