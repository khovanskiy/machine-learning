package ru.ifmo.ctddev.ml.knn;

import joinery.DataFrame;
import org.junit.Test;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.gui.beans.ScatterPlotMatrix;

import java.io.File;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

/**
 * @author victor
 */
public class KNNTest {

    @Test
    public void dataset1() throws Exception {
        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setFieldSeparator(";");
        csvLoader.setSource(new File("resources/dataset_1.csv"));
        Instances data = csvLoader.getDataSet();

        NumericToNominal convert = new NumericToNominal();
        //convert.setAttributeIndices("class");
        String[] options= new String[2];
        options[0]="-R";
        options[1]="last";  //range of variables to make numeric

        convert.setOptions(options);
        convert.setInputFormat(data);

        data = Filter.useFilter(data, convert);
        //data.deleteAttributeAt(1);
        //data.deleteAttributeAt(1);
        data.setClassIndex(2);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("resources/dataset_1.arff"));
        saver.writeBatch();

        //System.out.println(csvLoader.getStructure());
        Collections.shuffle(data);


        double m_x = 0.0;
        double m_y = 0.0;
        int k = 0;
        for (Instance instance : data) {
            if (((int) instance.value(2)) == 1) {
                m_x += instance.value(0);
                m_y += instance.value(1);
                ++k;
            }
        }
        m_x /= k;
        m_y /= k;

        //data.instance(0).

        float p = 0.8f;
        int length = (int) (data.size() * p);
        Instances train = new Instances(data, 0, length);
        Instances test = new Instances(data, length, data.size() - length);

        DataFrame<Object> df = new DataFrame<>();

        Classifier classifier = new IBk() {
            {
                setKNN(5);
            }
        };
        classifier.buildClassifier(train);
        //ScatterPlotMatrix
        Evaluation evaluation = new Evaluation(train);
        //evaluation.crossValidateModel(classifier, data, 2, new Random());
        evaluation.evaluateModel(classifier, test);
        //Ranker ranker = new Ranker();
        //GainRatioAttributeEval eval = new GainRatioAttributeEval();
        //eval.buildEvaluator(data);

        /*PrincipalComponents components = new PrincipalComponents();
        components.buildEvaluator(data);

        double[][] m = components.getCorrelationMatrix();
        for (int i = 0; i < m.length; ++i) {
            for (int j = i + 1; j < m[i].length; ++j) {
                if (i != j) {
                    if (Math.abs(m[i][j]) >= 0.3) {
                        System.out.println(m[i][j] + " " + data.attribute(i) + " " + data.attribute(j));
                    }
                }
            }
        }*/

        //Matrix.
        //System.out.println(components);
        /*for (int i = 0; i < data.numAttributes(); ++i) {
            System.out.println(data.attribute(i) + ": " + eval.evaluateAttribute(i));
        }*/

        //evaluation.evaluateModel(classifier, test);

        System.out.println(evaluation.toSummaryString("\nResult:\n======\n", false));
    }

    @Test
    public void dataset2() {

    }
}
