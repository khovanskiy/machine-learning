package ru.ifmo.ctddev.ml.nn;

import smile.classification.SVM;
import smile.math.kernel.LinearKernel;

import java.io.*;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
public class Serialize implements Runnable {
    public static void main(String[] args) throws IOException {
        new Serialize().run();
    }

    public static <T> T readObject(final File file) throws IOException, ClassNotFoundException {
        try (final FileInputStream fileStream = new FileInputStream(file);
             final ObjectInputStream objectStream = new ObjectInputStream(fileStream)) {
            return (T) objectStream.readObject();
        }
    }

    private static void writeObject(final File file, final Object object) throws IOException {
        try (final FileOutputStream fileStream = new FileOutputStream(file);
             final ObjectOutputStream objectStream = new ObjectOutputStream(fileStream)) {
            objectStream.writeObject(object);
        }
    }

    @Override
    public void run() {
        SVM<double[]> expected = new SVM<>(new LinearKernel(), 1);
        expected.learn(new double[]{0, 0}, 0);
        expected.learn(new double[]{0, 0}, 1);
        expected.learn(new double[]{0, 0}, 1);
        expected.learn(new double[]{1, 1}, 0);
        final File file = new File("./temp.out");
        try {
            writeObject(file, expected);
            System.out.println(expected);
            SVM<double[]> actual = readObject(file);
            System.out.println(actual);
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
