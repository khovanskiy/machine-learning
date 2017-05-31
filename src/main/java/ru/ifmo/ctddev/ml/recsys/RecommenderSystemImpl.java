package ru.ifmo.ctddev.ml.recsys;

import lombok.extern.slf4j.Slf4j;
import ru.ifmo.ctddev.ml.util.RandomUtils;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import static ru.ifmo.ctddev.ml.LinearUtils.dotProduct;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
@Slf4j
public class RecommenderSystemImpl implements RecommenderSystem {
    private final int featureCount;
    private final double gamma;
    private final double lambda;
    private final int iterationCount;
    private final double eps;

    private final Map<Long, Double> buMap = new HashMap<>();
    private final Map<Long, Double> biMap = new HashMap<>();
    private final Map<Long, double[]> puMap = new HashMap<>();
    private final Map<Long, double[]> qiMap = new HashMap<>();

    private double averageRating;

    public RecommenderSystemImpl(int featureCount, double gamma, double lambda, int iterationCount, double eps) {
        this.featureCount = featureCount;
        this.gamma = gamma;
        this.lambda = lambda;
        this.iterationCount = iterationCount;
        this.eps = eps;
    }


    @Override
    public Rating predict(long userId, long itemId) {
        double predicted = averageRating + getBu(userId) + getBi(itemId) + dotProduct(getPu(userId), getQi(itemId));
        return new Rating(userId, itemId, predicted);
    }

    @Override
    public void learn(Collection<? extends Rating> ratings) {
        log.info("eps      = " + eps);
        log.info("gamma    = " + gamma);
        log.info("lambda   = " + lambda);

        averageRating = ratings.stream().mapToDouble(Rating::getValue).average().orElse(0);

        int iteration = 0;
        double previousRmse = 0;
        double rmse = 1;
        while (iteration < iterationCount && Math.abs(previousRmse - rmse) > eps) {
            log.info("Iteration #{}", iteration);
            previousRmse = rmse;
            for (Rating rating : ratings) {

                double bu = getBu(rating.getUserId());
                double bi = getBi(rating.getItemId());

                double[] pu = getPu(rating.getUserId());
                double[] qi = getQi(rating.getItemId());

                double predicted = averageRating + bu + bi + dotProduct(qi, pu);
                double actual = rating.getValue();

                double error = actual - predicted;

                rmse += error * error;

                updateBu(rating.getUserId(), bu, error);
                updateBi(rating.getItemId(), bi, error);
                updatePuAndQi(pu, qi, error);
            }
            rmse = Math.sqrt(rmse / ratings.size());
            log.info("RMSE = " + rmse);
            ++iteration;
        }
    }

    private double[] getQi(long itemId) {
        return qiMap.computeIfAbsent(itemId, k -> randArray());
    }

    private double[] getPu(long userId) {
        return puMap.computeIfAbsent(userId, k -> randArray());
    }

    private Double getBi(long itemId) {
        return biMap.getOrDefault(itemId, 0.D);
    }

    private Double getBu(long userId) {
        return buMap.getOrDefault(userId, 0.D);
    }

    private void updatePuAndQi(double[] pu, double[] qi, double error) {
        for (int i = 0; i < featureCount; ++i) {
            double a = qi[i];
            double b = pu[i];
            qi[i] += gamma * (error * b - lambda * a);
            pu[i] += gamma * (error * a - lambda * b);
        }
    }

    private void updateBu(long userId, double bu, double error) {
        buMap.put(userId, bu + gamma * (error - lambda * bu));
    }

    private void updateBi(long itemId, double bi, double error) {
        biMap.put(itemId, bi + gamma * (error - lambda * bi));
    }

    private double[] randArray() {
        double[] array = new double[featureCount];
        for (int i = 0; i < featureCount; ++i) {
            array[i] = RandomUtils.randRange(-1.0 / (double) featureCount, 1.0 / (double) featureCount);
        }
        return array;
    }
}
