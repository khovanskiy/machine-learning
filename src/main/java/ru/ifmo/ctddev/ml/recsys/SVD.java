package ru.ifmo.ctddev.ml.recsys;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * @author victor
 */
public class SVD {
    private final static Random random = new Random();
    private final static int ITERATION_COUNT = 100;
    private final static double EPS = 1e-6;
    private final static double GAMMA = 0.005D;
    private final static double LAMBDA = 0.02D;

    /**
     * Отклонение пользователя u от среднего
     */
    private final Map<Long, Double> bu = new HashMap<>();
    /**
     * Отклонение объекта i от среднего
     */
    private final Map<Long, Double> bi = new HashMap<>();

    private final Map<Long, double[]> pu = new HashMap<>();
    private final Map<Long, double[]> qi = new HashMap<>();

    private final int f;
    private double mu;

    public SVD(int f) {
        this.f = f;
    }

    public void learn(Collection<? extends Mark> marks) {
        marks.forEach(mark -> {
            long user = mark.getUser();
            long item = mark.getItem();
            if (!pu.containsKey(user)) {
                pu.put(user, randArray());
            }
            if (!qi.containsKey(item)) {
                qi.put(item, randArray());
            }
            bu.putIfAbsent(user, 0.D);
            bi.putIfAbsent(item, 0.D);
        });
        mu = marks.stream().mapToDouble(Mark::getValue).average().getAsDouble();

        int iteration = 0;
        double prevRmse = 0;
        double rmse = 1;

        while (iteration < ITERATION_COUNT && Math.abs(prevRmse - rmse) > EPS) {
            prevRmse = rmse;

            for (Mark mark : marks) {
                long user = mark.getUser();
                long item = mark.getItem();
                long value = mark.getValue();

                double cbu = bu.get(user);
                double[] cpu = pu.get(user);
                double cbi = bi.get(item);
                double[] cqi = qi.get(item);

                double predict = mu + cbi + cbu + scalar(cpu, cqi);
                double error = value - predict;

                rmse += error * error;
                bu.put(user, cbu + GAMMA * (error - LAMBDA * cbu));
                bi.put(item, cbi + GAMMA * (error - LAMBDA * cbi));
                for (int i = 0; i < f; i++) {
                    double qi = cqi[i], pu = cpu[i];
                    cqi[i] = qi + GAMMA * (error * pu - LAMBDA * qi);
                    cpu[i] = pu + GAMMA * (error * qi - LAMBDA * pu);
                }

                rmse = Math.sqrt(rmse / marks.size());
            }

            ++iteration;
        }
    }

    private double[] randArray() {
        double[] array = new double[f];
        for (int i = 0; i < f; ++i) {
            array[i] = random.nextDouble();
        }
        return array;
    }

    private double scalar(double[] lhs, double[] rhs) {
        double res = 0;
        for (int i = 0; i < f; i++) {
            res += lhs[i] * rhs[i];
        }
        return res;
    }

    public double ask(long user, long item) {
        return mu + bu.getOrDefault(user, 0.D) + bi.getOrDefault(item, 0.D) + scalar(pu.getOrDefault(user, new double[f]), qi.getOrDefault(item, new double[f]));
    }
}
