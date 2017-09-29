package ru.ifmo.ctddev.ml.nn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.classification.ClassifierTrainer;
import smile.classification.OnlineClassifier;
import smile.classification.SoftClassifier;
import smile.math.Math;

import java.io.Serializable;

public class MyNeuralNetwork implements OnlineClassifier<double[]>, SoftClassifier<double[]>, Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = LoggerFactory.getLogger(MyNeuralNetwork.class);
    /**
     * The type of error function of network.
     */
    private ErrorFunction errorFunction = ErrorFunction.LEAST_MEAN_SQUARES;
    /**
     * The type of activation function in output layer.
     */
    private ActivationFunction activationFunction = ActivationFunction.LOGISTIC_SIGMOID;
    /**
     * The dimensionality of data.
     */
    private int p;
    /**
     * The number of classes.
     */
    private int k;
    /**
     * layers of this net
     */
    private Layer[] net;
    /**
     * input layer
     */
    private Layer inputLayer;
    /**
     * output layer
     */
    private Layer outputLayer;
    /**
     * learning rate
     */
    private double eta = 0.1;
    /**
     * momentum factor
     */
    private double alpha = 0.0;
    /**
     * weight decay factor, which is also a regularization term.
     */
    private double lambda = 0.0;
    /**
     * The buffer to store target value of training instance.
     */
    private double[] target;

    /**
     * Constructor. The activation function of output layer will be chosen
     * by natural pairing based on the error function and the number of
     * classes.
     *
     * @param error    the error function.
     * @param numUnits the number of units in each layer.
     */
    public MyNeuralNetwork(ErrorFunction error, int... numUnits) {
        this(error, natural(error, numUnits[numUnits.length - 1]), numUnits);
    }

    public MyNeuralNetwork(ErrorFunction error, ActivationFunction activation, Layer... layers) {
        int numLayers = layers.length;
        if (numLayers < 2) {
            throw new IllegalArgumentException("Invalid number of layers: " + numLayers);
        }
        this.net = layers;
        inputLayer = net[0];
        outputLayer = net[numLayers - 1];

        if (error == ErrorFunction.LEAST_MEAN_SQUARES) {
            if (activation == ActivationFunction.SOFTMAX) {
                throw new IllegalArgumentException("Sofmax activation function is invalid for least mean squares error.");
            }
        }

        if (error == ErrorFunction.CROSS_ENTROPY) {
            if (activation == ActivationFunction.LINEAR) {
                throw new IllegalArgumentException("Linear activation function is invalid with cross entropy error.");
            }

            if (activation == ActivationFunction.SOFTMAX && layers[numLayers - 1].units == 1) {
                throw new IllegalArgumentException("Softmax activation function is for multi-class.");
            }

            if (activation == ActivationFunction.LOGISTIC_SIGMOID && layers[numLayers - 1].units != 1) {
                throw new IllegalArgumentException("For cross entropy error, logistic sigmoid output is for binary classification.");
            }
        }

        this.errorFunction = error;
        this.activationFunction = activation;
    }

    /**
     * Constructor.
     *
     * @param error      the error function.
     * @param activation the activation function of output layer.
     * @param numUnits   the number of units in each layer.
     */
    public MyNeuralNetwork(ErrorFunction error, ActivationFunction activation, int... numUnits) {
        int numLayers = numUnits.length;
        if (numLayers < 2) {
            throw new IllegalArgumentException("Invalid number of layers: " + numLayers);
        }

        for (int i = 0; i < numLayers; i++) {
            if (numUnits[i] < 1) {
                throw new IllegalArgumentException(String.format("Invalid number of units of layer %d: %d", i + 1, numUnits[i]));
            }
        }

        if (error == ErrorFunction.LEAST_MEAN_SQUARES) {
            if (activation == ActivationFunction.SOFTMAX) {
                throw new IllegalArgumentException("Sofmax activation function is invalid for least mean squares error.");
            }
        }

        if (error == ErrorFunction.CROSS_ENTROPY) {
            if (activation == ActivationFunction.LINEAR) {
                throw new IllegalArgumentException("Linear activation function is invalid with cross entropy error.");
            }

            if (activation == ActivationFunction.SOFTMAX && numUnits[numLayers - 1] == 1) {
                throw new IllegalArgumentException("Softmax activation function is for multi-class.");
            }

            if (activation == ActivationFunction.LOGISTIC_SIGMOID && numUnits[numLayers - 1] != 1) {
                throw new IllegalArgumentException("For cross entropy error, logistic sigmoid output is for binary classification.");
            }
        }

        this.errorFunction = error;
        this.activationFunction = activation;

        if (error == ErrorFunction.CROSS_ENTROPY) {
            this.alpha = 0.0;
            this.lambda = 0.0;
        }

        this.p = numUnits[0];
        this.k = numUnits[numLayers - 1] == 1 ? 2 : numUnits[numLayers - 1];
        this.target = new double[numUnits[numLayers - 1]];

        net = new Layer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            net[i] = new Layer();
            net[i].units = numUnits[i];
            net[i].output = new double[numUnits[i] + 1];
            net[i].error = new double[numUnits[i] + 1];
            net[i].output[numUnits[i]] = 1.0;
        }

        inputLayer = net[0];
        outputLayer = net[numLayers - 1];

        // Initialize random weights.
        for (int l = 1; l < numLayers; l++) {
            net[l].weight = new double[numUnits[l]][numUnits[l - 1] + 1];
            net[l].delta = new double[numUnits[l]][numUnits[l - 1] + 1];
            double r = 1.0 / Math.sqrt(net[l - 1].units);
            for (int i = 0; i < net[l].units; i++) {
                for (int j = 0; j <= net[l - 1].units; j++) {
                    net[l].weight[i][j] = Math.random(-r, r);
                }
            }
        }
    }

    /**
     * Private constructor for clone purpose.
     */
    private MyNeuralNetwork() {

    }

    /**
     * Returns the activation function of output layer based on natural pairing.
     *
     * @param error the error function.
     * @param k     the number of output nodes.
     * @return the activation function of output layer based on natural pairing
     */
    private static ActivationFunction natural(ErrorFunction error, int k) {
        if (error == ErrorFunction.CROSS_ENTROPY) {
            if (k == 1) {
                return ActivationFunction.LOGISTIC_SIGMOID;
            } else {
                return ActivationFunction.SOFTMAX;
            }
        } else {
            return ActivationFunction.LOGISTIC_SIGMOID;
        }

    }

    /**
     * Returns natural log without underflow.
     */
    private static double log(double x) {
        double y = 0.0;
        if (x < 1E-300) {
            y = -690.7755;
        } else {
            y = Math.log(x);
        }
        return y;
    }

    @Override
    public MyNeuralNetwork clone() {
        MyNeuralNetwork copycat = new MyNeuralNetwork();

        copycat.errorFunction = errorFunction;
        copycat.activationFunction = activationFunction;
        copycat.p = p;
        copycat.k = k;
        copycat.eta = eta;
        copycat.alpha = alpha;
        copycat.lambda = lambda;
        copycat.target = target.clone();

        int numLayers = net.length;
        copycat.net = new Layer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            copycat.net[i] = new Layer();
            copycat.net[i].units = net[i].units;
            copycat.net[i].output = net[i].output.clone();
            copycat.net[i].error = net[i].error.clone();
            if (i > 0) {
                copycat.net[i].weight = Math.clone(net[i].weight);
                copycat.net[i].delta = Math.clone(net[i].delta);
            }
        }

        copycat.inputLayer = copycat.net[0];
        copycat.outputLayer = copycat.net[numLayers - 1];

        return copycat;
    }

    /**
     * Returns the learning rate.
     */
    public double getLearningRate() {
        return eta;
    }

    /**
     * Sets the learning rate.
     *
     * @param eta the learning rate.
     */
    public void setLearningRate(double eta) {
        if (eta <= 0) {
            throw new IllegalArgumentException("Invalid learning rate: " + eta);
        }
        this.eta = eta;
    }

    /**
     * Returns the momentum factor.
     */
    public double getMomentum() {
        return alpha;
    }

    /**
     * Sets the momentum factor.
     *
     * @param alpha the momentum factor.
     */
    public void setMomentum(double alpha) {
        if (alpha < 0.0 || alpha >= 1.0) {
            throw new IllegalArgumentException("Invalid momentum factor: " + alpha);
        }

        this.alpha = alpha;
    }

    /**
     * Returns the weight decay factor.
     */
    public double getWeightDecay() {
        return lambda;
    }

    /**
     * Sets the weight decay factor. After each weight update, every weight
     * is simply ''decayed'' or shrunk according w = w * (1 - eta * lambda).
     *
     * @param lambda the weight decay for regularization.
     */
    public void setWeightDecay(double lambda) {
        if (lambda < 0.0 || lambda > 0.1) {
            throw new IllegalArgumentException("Invalid weight decay factor: " + lambda);
        }

        this.lambda = lambda;
    }

    /**
     * Sets the input vector into the input layer.
     *
     * @param x the input vector.
     */
    private void setInput(double[] x) {
        if (x.length != inputLayer.units) {
            throw new IllegalArgumentException(String.format("Invalid input vector size: %d, expected: %d", x.length, inputLayer.units));
        }
        System.arraycopy(x, 0, inputLayer.output, 0, inputLayer.units);
    }

    /**
     * Returns the output vector into the given array.
     *
     * @param y the output vector.
     */
    private void getOutput(double[] y) {
        if (y.length != outputLayer.units) {
            throw new IllegalArgumentException(String.format("Invalid output vector size: %d, expected: %d", y.length, outputLayer.units));
        }
        System.arraycopy(outputLayer.output, 0, y, 0, outputLayer.units);
    }

    /**
     * Propagates signals from a lower layer to the next upper layer.
     *
     * @param lower the lower layer where signals are from.
     * @param upper the upper layer where signals are propagated to.
     */
    private void propagate(Layer lower, Layer upper) {
        for (int i = 0; i < upper.units; i++) {
            double sum = 0.0;
            for (int j = 0; j <= lower.units; j++) {
                sum += upper.weight[i][j] * lower.output[j];
            }

            if (upper != outputLayer || activationFunction == ActivationFunction.LOGISTIC_SIGMOID) {
                upper.output[i] = Math.logistic(sum);
            } else {
                if (activationFunction == ActivationFunction.LINEAR || activationFunction == ActivationFunction.SOFTMAX) {
                    upper.output[i] = sum;
                } else {
                    throw new UnsupportedOperationException("Unsupported activation function.");
                }
            }
        }

        if (upper == outputLayer && activationFunction == ActivationFunction.SOFTMAX) {
            softmax();
        }
    }

    /**
     * Calculate softmax activation function in output layer without overflow.
     */
    private void softmax() {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < outputLayer.units; i++) {
            if (outputLayer.output[i] > max) {
                max = outputLayer.output[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < outputLayer.units; i++) {
            double out = Math.exp(outputLayer.output[i] - max);
            outputLayer.output[i] = out;
            sum += out;
        }

        for (int i = 0; i < outputLayer.units; i++) {
            outputLayer.output[i] /= sum;
        }
    }

    /**
     * Propagates the signals through the neural network.
     */
    private void propagate() {
        for (int l = 0; l < net.length - 1; l++) {
            propagate(net[l], net[l + 1]);
        }
    }

    /**
     * Compute the network output error.
     *
     * @param output the desired output.
     */
    private double computeOutputError(double[] output) {
        return computeOutputError(output, outputLayer.error);
    }

    /**
     * Compute the network output error.
     *
     * @param output   the desired output.
     * @param gradient the array to store gradient on output.
     * @return the error defined by loss function.
     */
    private double computeOutputError(double[] output, double[] gradient) {
        if (output.length != outputLayer.units) {
            throw new IllegalArgumentException(String.format("Invalid output vector size: %d, expected: %d", output.length, outputLayer.units));
        }

        double error = 0.0;
        for (int i = 0; i < outputLayer.units; i++) {
            double out = outputLayer.output[i];
            double g = output[i] - out;

            if (errorFunction == ErrorFunction.LEAST_MEAN_SQUARES) {
                error += 0.5 * g * g;
            } else if (errorFunction == ErrorFunction.CROSS_ENTROPY) {
                if (activationFunction == ActivationFunction.SOFTMAX) {
                    error -= output[i] * log(out);
                } else if (activationFunction == ActivationFunction.LOGISTIC_SIGMOID) {
                    // We have only one output neuron in this case.
                    error = -output[i] * log(out) - (1.0 - output[i]) * log(1.0 - out);
                }
            }

            if (errorFunction == ErrorFunction.LEAST_MEAN_SQUARES && activationFunction == ActivationFunction.LOGISTIC_SIGMOID) {
                g *= out * (1.0 - out);
            }

            gradient[i] = g;
        }

        return error;
    }

    /**
     * Propagates the errors back from a upper layer to the next lower layer.
     *
     * @param upper the lower layer where errors are from.
     * @param lower the upper layer where errors are propagated back to.
     */
    private void backpropagate(Layer upper, Layer lower) {
        for (int i = 0; i <= lower.units; i++) {
            double out = lower.output[i];
            double err = 0;
            for (int j = 0; j < upper.units; j++) {
                err += upper.weight[j][i] * upper.error[j];
            }
            lower.error[i] = out * (1.0 - out) * err;
        }
    }

    /**
     * Propagates the errors back through the network.
     */
    private void backpropagate() {
        for (int l = net.length; --l > 0; ) {
            backpropagate(net[l], net[l - 1]);
        }
    }

    /**
     * Adjust network weights by back-propagation algorithm.
     */
    private void adjustWeights() {
        for (int l = 1; l < net.length; l++) {
            for (int i = 0; i < net[l].units; i++) {
                for (int j = 0; j <= net[l - 1].units; j++) {
                    double out = net[l - 1].output[j];
                    double err = net[l].error[i];
                    double delta = (1 - alpha) * eta * err * out + alpha * net[l].delta[i][j];
                    net[l].delta[i][j] = delta;
                    net[l].weight[i][j] += delta;
                    if (lambda != 0.0 && j < net[l - 1].units) {
                        net[l].weight[i][j] *= (1.0 - eta * lambda);
                    }
                }
            }
        }
    }

    /**
     * Predict the target value of a given instance. Note that this method is NOT
     * multi-thread safe.
     *
     * @param x the instance.
     * @param y the array to store network output on output. For softmax
     *          activation function, these are estimated posteriori probabilities.
     * @return the predicted class label.
     */
    @Override
    public int predict(double[] x, double[] y) {
        setInput(x);
        propagate();
        getOutput(y);

        if (outputLayer.units == 1) {
            if (outputLayer.output[0] > 0.5) {
                return 0;
            } else {
                return 1;
            }
        }

        double max = Double.NEGATIVE_INFINITY;
        int label = -1;
        for (int i = 0; i < outputLayer.units; i++) {
            if (outputLayer.output[i] > max) {
                max = outputLayer.output[i];
                label = i;
            }
        }
        return label;
    }

    /**
     * Predict the class of a given instance. Note that this method is NOT
     * multi-thread safe.
     *
     * @param x the instance.
     * @return the predicted class label.
     */
    @Override
    public int predict(double[] x) {
        setInput(x);
        propagate();

        if (outputLayer.units == 1) {
            if (outputLayer.output[0] > 0.5) {
                return 0;
            } else {
                return 1;
            }
        }

        double max = Double.NEGATIVE_INFINITY;
        int label = -1;
        for (int i = 0; i < outputLayer.units; i++) {
            if (outputLayer.output[i] > max) {
                max = outputLayer.output[i];
                label = i;
            }
        }
        return label;
    }

    /**
     * Update the neural network with given instance and associated target value.
     * Note that this method is NOT multi-thread safe.
     *
     * @param x      the training instance.
     * @param y      the target value.
     * @param weight a positive weight value associated with the training instance.
     * @return the weighted training error before back-propagation.
     */
    public double learn(double[] x, double[] y, double weight) {
        setInput(x);
        propagate();

        double err = weight * computeOutputError(y);

        if (weight != 1.0) {
            for (int i = 0; i < outputLayer.units; i++) {
                outputLayer.error[i] *= weight;
            }
        }

        backpropagate();
        adjustWeights();
        return err;
    }

    @Override
    public void learn(double[] x, int y) {
        learn(x, y, 1.0);
    }

    /**
     * Online update the neural network with a new training instance.
     * Note that this method is NOT multi-thread safe.
     *
     * @param x      training instance.
     * @param y      training label.
     * @param weight a positive weight value associated with the training instance.
     */
    public void learn(double[] x, int y, double weight) {
        if (weight < 0.0) {
            throw new IllegalArgumentException("Invalid weight: " + weight);
        }

        if (weight == 0.0) {
            logger.info("Ignore the training instance with zero weight.");
            return;
        }

        if (y < 0) {
            throw new IllegalArgumentException("Invalid class label: " + y);
        }

        if (outputLayer.units == 1 && y > 1) {
            throw new IllegalArgumentException("Invalid class label: " + y);
        }

        if (outputLayer.units > 1 && y >= outputLayer.units) {
            throw new IllegalArgumentException("Invalid class label: " + y);
        }

        if (errorFunction == ErrorFunction.CROSS_ENTROPY) {
            if (activationFunction == ActivationFunction.LOGISTIC_SIGMOID) {
                if (y == 0) {
                    target[0] = 1.0;
                } else {
                    target[0] = 0.0;
                }
            } else {
                for (int i = 0; i < target.length; i++) {
                    target[i] = 0.0;
                }
                target[y] = 1.0;
            }
        } else {
            for (int i = 0; i < target.length; i++) {
                target[i] = 0.1;
            }
            target[y] = 0.9;
        }

        learn(x, target, weight);
    }

    /**
     * Trains the neural network with the given dataset for one epoch by
     * stochastic gradient descent.
     *
     * @param x training instances.
     * @param y training labels in [0, k), where k is the number of classes.
     */
    public void learn(double[][] x, int[] y) {
        int n = x.length;
        int[] index = Math.permutate(n);
        for (int i = 0; i < n; i++) {
            learn(x[index[i]], y[index[i]]);
        }
    }

    /**
     * The types of error functions.
     */
    public enum ErrorFunction {
        /**
         * Least mean squares error function.
         */
        LEAST_MEAN_SQUARES,

        /**
         * Cross entropy error function for output as probabilities.
         */
        CROSS_ENTROPY
    }

    /**
     * The types of activation functions in output layer. In this implementation,
     * the hidden layers always employs logistic sigmoid activation function.
     */
    public enum ActivationFunction {
        /**
         * Linear activation function.
         */
        LINEAR,

        /**
         * Logistic sigmoid activation function. For multi-class classification,
         * each unit in output layer corresponds to a class. For binary
         * classification and cross entropy error function, there is only
         * one output unit whose value can be regarded as posteriori probability.
         */
        LOGISTIC_SIGMOID,

        /**
         * Softmax activation for multi-class cross entropy objection function.
         * The values of units in output layer can be regarded as posteriori
         * probabilities of each class.
         */
        SOFTMAX
    }

    /**
     * Trainer for neural networks.
     */
    public static class Trainer extends ClassifierTrainer<double[]> {
        /**
         * The type of error function of network.
         */
        private ErrorFunction errorFunction = ErrorFunction.LEAST_MEAN_SQUARES;
        /**
         * The type of activation function in output layer.
         */
        private ActivationFunction activationFunction = ActivationFunction.LOGISTIC_SIGMOID;
        /**
         * The number of units in each layer.
         */
        private int[] numUnits;
        /**
         * learning rate
         */
        private double eta = 0.1;
        /**
         * momentum factor
         */
        private double alpha = 0.0;
        /**
         * weight decay factor, which is also a regularization term.
         */
        private double lambda = 0.0;
        /**
         * The number of epochs of stochastic learning.
         */
        private int epochs = 25;

        /**
         * Constructor. The activation function of output layer will be chosen
         * by natural pairing based on the error function and the number of
         * classes.
         *
         * @param error    the error function.
         * @param numUnits the number of units in each layer.
         */
        public Trainer(ErrorFunction error, int... numUnits) {
            this(error, natural(error, numUnits[numUnits.length - 1]), numUnits);
        }

        /**
         * Constructor.
         *
         * @param error      the error function.
         * @param activation the activation function of output layer.
         * @param numUnits   the number of units in each layer.
         */
        public Trainer(ErrorFunction error, ActivationFunction activation, int... numUnits) {
            int numLayers = numUnits.length;
            if (numLayers < 2) {
                throw new IllegalArgumentException("Invalid number of layers: " + numLayers);
            }

            for (int i = 0; i < numLayers; i++) {
                if (numUnits[i] < 1) {
                    throw new IllegalArgumentException(String.format("Invalid number of units of layer %d: %d", i + 1, numUnits[i]));
                }
            }

            if (error == ErrorFunction.LEAST_MEAN_SQUARES) {
                if (activation == ActivationFunction.SOFTMAX) {
                    throw new IllegalArgumentException("Sofmax activation function is invalid for least mean squares error.");
                }
            }

            if (error == ErrorFunction.CROSS_ENTROPY) {
                if (activation == ActivationFunction.LINEAR) {
                    throw new IllegalArgumentException("Linear activation function is invalid with cross entropy error.");
                }

                if (activation == ActivationFunction.SOFTMAX && numUnits[numLayers - 1] == 1) {
                    throw new IllegalArgumentException("Softmax activation function is for multi-class.");
                }

                if (activation == ActivationFunction.LOGISTIC_SIGMOID && numUnits[numLayers - 1] != 1) {
                    throw new IllegalArgumentException("For cross entropy error, logistic sigmoid output is for binary classification.");
                }
            }

            this.errorFunction = error;
            this.activationFunction = activation;
            this.numUnits = numUnits;
        }

        /**
         * Sets the learning rate.
         *
         * @param eta the learning rate.
         */
        public Trainer setLearningRate(double eta) {
            if (eta <= 0) {
                throw new IllegalArgumentException("Invalid learning rate: " + eta);
            }
            this.eta = eta;
            return this;
        }

        /**
         * Sets the momentum factor.
         *
         * @param alpha the momentum factor.
         */
        public Trainer setMomentum(double alpha) {
            if (alpha < 0.0 || alpha >= 1.0) {
                throw new IllegalArgumentException("Invalid momentum factor: " + alpha);
            }

            this.alpha = alpha;
            return this;
        }

        /**
         * Sets the weight decay factor. After each weight update, every weight
         * is simply ''decayed'' or shrunk according w = w * (1 - eta * lambda).
         *
         * @param lambda the weight decay for regularization.
         */
        public Trainer setWeightDecay(double lambda) {
            if (lambda < 0.0 || lambda > 0.1) {
                throw new IllegalArgumentException("Invalid weight decay factor: " + lambda);
            }

            this.lambda = lambda;
            return this;
        }

        /**
         * Sets the number of epochs of stochastic learning.
         *
         * @param epochs the number of epochs of stochastic learning.
         */
        public Trainer setNumEpochs(int epochs) {
            if (epochs < 1) {
                throw new IllegalArgumentException("Invalid numer of epochs of stochastic learning:" + epochs);
            }

            this.epochs = epochs;
            return this;
        }

        @Override
        public MyNeuralNetwork train(double[][] x, int[] y) {
            MyNeuralNetwork oldNet = null;
            MyNeuralNetwork currentNet = null;

            int currentLayer = 0;
            while (currentLayer < numUnits.length - 2) {
                logger.info("Learn the {} layer", currentLayer + 1);
                int[] units = new int[currentLayer + 2];
                for (int i = 0; i < currentLayer + 1; ++i) {
                    units[i] = numUnits[i];
                }
                units[units.length - 1] = numUnits[numUnits.length - 1];
                currentNet = new MyNeuralNetwork(errorFunction, activationFunction, units);
                if (oldNet != null) {
                    currentNet.inputLayer = oldNet.inputLayer;
                    for (int i = 0; i < currentLayer + 1; ++i) {
                        currentNet.net[i] = oldNet.net[i];
                    }
                }
                currentNet.setLearningRate(eta);
                currentNet.setMomentum(alpha);
                currentNet.setWeightDecay(lambda);

                for (int i = 1; i <= epochs; i++) {
                    currentNet.learn(x, y);
                    logger.info("Neural network learns epoch {}", i);
                }
                ++currentLayer;
                oldNet = currentNet;
            }

            return currentNet;
        }
    }

    /**
     * A layer of a feed forward neural network.
     */
    private class Layer implements Serializable {
        private static final long serialVersionUID = 1L;

        /**
         * number of units in this layer
         */
        int units;
        /**
         * output of i<i>th</i> unit
         */
        double[] output;
        /**
         * error term of i<i>th</i> unit
         */
        double[] error;
        /**
         * connection weights to i<i>th</i> unit from previous layer
         */
        double[][] weight;
        /**
         * last weight changes for momentum
         */
        double[][] delta;
    }
}
