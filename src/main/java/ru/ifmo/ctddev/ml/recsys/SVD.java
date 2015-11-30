package ru.ifmo.ctddev.ml.recsys;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;

/**
 * @author victor
 */
public class SVD {
    private final static Random random = new Random();
    private final static int ITERATION_COUNT = 100;
    private final static int JOG_OF_WEIGHTS_COUNT = 5;
    private final static double EPS = 1e-6;
    private final static double GAMMA = 0.005D;
    private final static double VICINITY = 0.00005D;
    private double lambda1 = 0.02D;
    //private double lambda2 = 0.015D;
    /**
     * The set R(u) contains the items rated by user u
     */
    private final Map<Long, Set<Long>> Ru = new HashMap<>();

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
    private final Map<Long, double[]> yi = new HashMap<>();

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
            if (!yi.containsKey(item)) {
                yi.put(item, randArray());
            }
            Set<Long> cRu = Ru.get(user);
            if (cRu == null) {
                cRu = new HashSet<>();
                Ru.put(user, cRu);
            }
            cRu.add(item);
            bu.putIfAbsent(user, 0.D);
            bi.putIfAbsent(item, 0.D);
        });
        mu = marks.stream().mapToDouble(Mark::getValue).average().getAsDouble();

        //lambda = 1.0 / (double) marks.size();

        for (int k = 0; k < JOG_OF_WEIGHTS_COUNT; ++k) {
            System.out.println("Jogging #" + k);
            if (k != 0) {
                for (Mark mark : marks) {
                    long user = mark.getUser();
                    long item = mark.getItem();

                    double cbu = bu.get(user);
                    cbu += randRange(-cbu * VICINITY, cbu * VICINITY);
                    bu.put(user, cbu);
                    double cbi = bi.get(item);
                    cbi += randRange(-cbi * VICINITY, cbi * VICINITY);
                    bi.put(item, cbi);

                    double[] cpu = pu.get(user);
                    for (int i = 0; i < cpu.length; ++i) {
                        double diff = randRange(-cpu[i] * VICINITY, cpu[i] * VICINITY);
                        cpu[i] += diff;
                    }
                    pu.put(user, cpu);
                    double[] cqi = qi.get(item);
                    for (int i = 0; i < cqi.length; ++i) {
                        double diff = randRange(-cqi[i] * VICINITY, cqi[i] * VICINITY);
                        cqi[i] += diff;
                    }
                    qi.put(item, cqi);
                }
            }
            int iteration = 0;
            double prevRmse = 0;
            double rmse = 1;
            while (iteration < ITERATION_COUNT && Math.abs(prevRmse - rmse) > EPS) {
                prevRmse = rmse;
                System.out.println("Iteration #" + iteration);
                for (Mark mark : marks) {
                    long user = mark.getUser();
                    long item = mark.getItem();
                    long value = mark.getValue();

                    double cbu = bu.get(user);
                    double[] cpu = pu.get(user);
                    double cbi = bi.get(item);
                    double[] cqi = qi.get(item);

                    Set<Long> cRu = Ru.get(user);
                    double[] cyi = yi.get(item);

                    double predict = mu + cbi + cbu + scalar(cpu, cqi);
                    double error = value - predict;

                    rmse += error * error;
                    bu.put(user, cbu + GAMMA * (error - lambda1 * cbu));
                    bi.put(item, cbi + GAMMA * (error - lambda1 * cbi));

                    /*assert cRu.size() > 0;
                    double[] koeff = new double[f];
                    for (long j : cRu) {
                        double[] yj = yi.get(j);
                        for (int i = 0; i < f; ++i) {
                            koeff[i] += yj[i];
                        }
                    }*/
                    /*for (int i = 0; i < f; ++i) {
                        koeff[i] *= Math.sqrt(cRu.size());
                    }*/
                    for (int i = 0; i < f; i++) {
                        double qi = cqi[i], pu = cpu[i];
                        cqi[i] = qi + GAMMA * (error * pu - lambda1 * qi);
                        cpu[i] = pu + GAMMA * (error * qi - lambda1 * pu);
                    }
                    /*for (long j : cRu) {
                        double[] cyj = yi.get(j);
                        for (int i = 0; i < f; ++i) {
                            double qi = cqi[i];
                            double yj = cyj[i];
                            cyi[i] = yj + GAMMA * (error * Math.sqrt(cRu.size()) * qi - lambda2 * yj);
                        }
                    }*/

                    rmse = Math.sqrt(rmse / marks.size());
                }

                ++iteration;
            }
        }
        System.out.println("EPS      = " + EPS);
        System.out.println("GAMMA    = " + GAMMA);
        System.out.println("LAMBDA   = " + lambda1);
        System.out.println("VICINITY = " + VICINITY);
    }

    public static double randRange(double low, double high) {
        return random.nextDouble() * (high - low) + low;
    }

    private double[] randArray() {
        double[] array = new double[f];
        for (int i = 0; i < f; ++i) {
            //array[i] = random.nextDouble();
            array[i] = randRange(-1.0 / (double) f, 1.0 / (double) f);
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
