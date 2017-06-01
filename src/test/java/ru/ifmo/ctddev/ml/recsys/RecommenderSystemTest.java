package ru.ifmo.ctddev.ml.recsys;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
@Slf4j
public class RecommenderSystemTest {
    @Test
    public void test() throws IOException {
        int featureCount = 7;
        double gamma = 0.005;
        double lambda = 0.03;
        int iterationCount = 100;
        double eps = 1e-6;
        //final RecommenderSystem recommenderSystem = new RecommenderSystemImpl(featureCount, gamma, lambda, iterationCount, eps);
        final RecommenderSystem recommenderSystem = new RecommenderSystemStub(5);
        final List<File> train = Arrays.asList(
                new File("resources", "train.csv")
                //new File("resources", "validation.csv")
        );

        List<Rating> trainRatings = train.stream()
                .map(this::readRatings)
                .flatMap(Collection::stream)
                .collect(Collectors.toList());
        recommenderSystem.learn(trainRatings);

        validate(recommenderSystem);

        //generateSubmission(recommenderSystem);
    }

    private void validate(RecommenderSystem recommenderSystem) {
        log.info("Validation");
        final File validation = new File("resources", "validation.csv");
        List<Rating> validationRatings = readRatings(validation);
        double rmse = 0;

        int count = 0;
        for (Rating actual : validationRatings) {
            log.info("Validating #{}/{}", count, validationRatings.size());
            Rating predicted = recommenderSystem.predict(actual.getUserId(), actual.getItemId());
            //log.debug("User = " + actual.getUserId() + ", Item = " + actual.getItemId() + " | A = " + actual.getValue() + " P = " + predicted.getValue());
            rmse += Math.pow(predicted.getValue() - actual.getValue(), 2);
            ++count;
            if (count > 2500) {
                break;
            }
        }
        rmse = Math.sqrt(rmse / count);
        log.info("RMSE = " + rmse);
    }

    private void generateSubmission(RecommenderSystem recommenderSystem) throws IOException {
        final File testIds = new File("resources", "test-ids.csv");
        try (BufferedReader reader = new BufferedReader(new FileReader(testIds))) {
            reader.readLine();
            final File output = new File("output2.csv");
            try (PrintWriter writer = new PrintWriter(output)) {
                writer.println("id,rating");
                while (reader.ready()) {
                    String[] splices = reader.readLine().split(",");
                    long id = Long.parseLong(splices[0]);
                    long user = Long.parseLong(splices[1]);
                    long item = Long.parseLong(splices[2]);
                    writer.println(id + "," + recommenderSystem.predict(user, item).getValue());
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Reads the rating data from CSV file.
     *
     * @param input the file
     * @return list of ratings
     */
    private List<Rating> readRatings(File input) {
        List<Rating> ratings = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(input))) {
            reader.readLine(); // header
            while (reader.ready()) {
                String[] splices = reader.readLine().split(",");
                long userId = Long.parseLong(splices[0]);
                long itemId = Long.parseLong(splices[1]);
                double value = Double.parseDouble(splices[2]);
                ratings.add(new Rating(userId, itemId, value));
            }
        } catch (IOException ignored) {
        }
        return ratings;
    }
}
