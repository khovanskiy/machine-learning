package ru.ifmo.ctddev.ml.util;

import java.util.Random;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
public class RandomUtils {
    private static final Random random = new Random();

    public static double randRange(double low, double high) {
        return random.nextDouble() * (high - low) + low;
    }
}
