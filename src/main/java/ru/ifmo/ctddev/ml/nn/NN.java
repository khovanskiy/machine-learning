package ru.ifmo.ctddev.ml.nn;

import lombok.extern.slf4j.Slf4j;
import smile.validation.Precision;
import smile.validation.Validation;

import java.io.*;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
@Slf4j
public class NN implements Runnable {

    double[][] train;
    int[] labels;

    public static void main(String[] args) {
        new NN().run();
    }

    @Override
    public void run() {
        final File labelsFile = new File("./resources/train-labels.idx1-ubyte");
        final File imagesFile = new File("./resources/train-images.idx3-ubyte");
        final byte[] buffer = new byte[1024];
        try {
            openStream(labelsFile, labelsStream -> {
                int a = labelsStream.readInt();
                if (a != 2049) {
                    throw new RuntimeException("Label file has wrong magic number: " + a + " (should be 2049)");
                }
                int numLabels = labelsStream.readInt();
                openStream(imagesFile, imagesStream -> {
                    int magicNumber = imagesStream.readInt();
                    if (magicNumber != 2051) {
                        throw new RuntimeException("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
                    }
                    int numImages = imagesStream.readInt();

                    if (numLabels != numImages) {
                        final String str = "Image file and label file do not contain the same number of entries.\n" +
                                "  Label file contains: " + numLabels + "\n" +
                                "  Image file contains: " + numImages + "\n";
                        throw new RuntimeException(str);
                    }

                    int numRows = imagesStream.readInt();
                    int numCols = imagesStream.readInt();

                    train = new double[numImages][];
                    byte[] labelsData = new byte[numLabels];
                    int read;
                    int offset = 0;
                    while (offset < labelsData.length && (read = labelsStream.read(labelsData, offset, labelsData.length - offset)) != -1) {
                        offset += read;
                    }

                    labels = new int[numLabels];
                    for (int i = 0; i < labelsData.length; ++i) {
                        labels[i] = labelsData[i];
                    }

                    int imageVectorSize = numCols * numRows;
                    double[] image = new double[imageVectorSize];
                    int curImage = 0;
                    int imageIndex = 0;
                    while ((read = imagesStream.read(buffer, 0, buffer.length)) != -1) {
                        for (int i = 0; i < read; ++i) {
                            image[imageIndex] = (buffer[i] & 0xff) / 255d;
                            ++imageIndex;
                            if (imageIndex == imageVectorSize) {
                                imageIndex = 0;
                                //printImage(image, numRows, numCols);
                                train[curImage] = image;
                                image = new double[imageVectorSize];
                                ++curImage;
                            }
                        }
                    }
                });
            });


            log.info("Cross Validation...");
            /*double precision = Validation.cv(2, new SVM.CascadeTrainer<>(new LinearKernel(), 1, 10, SVM.Multiclass.ONE_VS_ONE), train, labels, new Precision());*/
            final int vectorSize = train[0].length;

            final int[] units = {vectorSize, vectorSize >> 1, vectorSize >> 2, vectorSize >> 3, 10};

            int folds = 2;
            int baseEpochs = 1;
            long timestamp = System.currentTimeMillis();
            double precision = Validation.cv(folds, new MyNeuralNetwork.Trainer(MyNeuralNetwork.ErrorFunction.CROSS_ENTROPY, units).setNumEpochs((int) (baseEpochs * 2.5f)), train, labels, new Precision());
            log.info("Precision = " + precision);
            log.info("Time = " + (System.currentTimeMillis() - timestamp));
            timestamp = System.currentTimeMillis();
            precision = Validation.cv(folds, new MyNeuralNetwork.CascadeTrainer(MyNeuralNetwork.ErrorFunction.CROSS_ENTROPY, units).setNumEpochs(baseEpochs), train, labels, new Precision());
            log.info("Precision = " + precision);
            log.info("Time = " + (System.currentTimeMillis() - timestamp));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printImage(double[] image, int numRows, int numCols) {
        int k = 0;
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                if (image[k] > 0) {
                    System.out.print("$");
                } else {
                    System.out.print(".");
                }
                ++k;
            }
            System.out.println();
        }
    }

    public void openStream(final File file, final IOConsumer<DataInputStream> consumer) throws IOException {
        try (FileInputStream fileStream = new FileInputStream(file);
             BufferedInputStream bufferedStream = new BufferedInputStream(fileStream);
             DataInputStream dataStream = new DataInputStream(bufferedStream)) {
            consumer.accept(dataStream);
        }
    }

    @FunctionalInterface
    public interface IOConsumer<T> {
        void accept(T t) throws IOException;
    }
}
