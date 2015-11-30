package ru.ifmo.ctddev.ml.recsys;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author victor
 */
public class RecSys implements Runnable {

    public static void main(String[] args) {
        new RecSys().run();
    }

    @Override
    public void run() {
        List<Mark> marks = Arrays.asList("train.csv", "validation.csv").stream().map(filename -> {
            List<Mark> temp = new ArrayList<>();
            File file = new File(filename);
            try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                reader.readLine(); // header
                while (reader.ready()) {
                    String[] splices = reader.readLine().split(",");
                    long user = Long.parseLong(splices[0]);
                    long item = Long.parseLong(splices[1]);
                    long value = Long.parseLong(splices[2]);
                    temp.add(new Mark(user, item, value));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return temp;
        }).flatMap(Collection::stream).collect(Collectors.toList());

        SVD svd = new SVD(7);
        svd.learn(marks);
        try (BufferedReader reader = new BufferedReader(new FileReader(new File("test-ids.csv")))) {
            reader.readLine();
            try (PrintWriter writer = new PrintWriter(new File("output.csv"))) {
                writer.println("id,rating");
                while (reader.ready()) {
                    String[] splices = reader.readLine().split(",");
                    long user = Long.parseLong(splices[1]);
                    long item = Long.parseLong(splices[2]);
                    writer.println(splices[0] + "," + svd.ask(user, item));
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
