package ru.ifmo.ctddev.ml;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
public class LinearUtils {
    public static double dotProduct(double[] a, double[] b) {
        double c = 0;
        for (int i = 0; i < a.length; ++i) {
            c += a[i] * b[i];
        }
        return c;
    }
}
