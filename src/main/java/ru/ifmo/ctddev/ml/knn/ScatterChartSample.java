package ru.ifmo.ctddev.ml.knn;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.Collections;


public class ScatterChartSample extends Application {

    @Override public void start(Stage stage) throws Exception {
        stage.setTitle("Scatter Chart Sample");
        final NumberAxis xAxis = new NumberAxis(-1, 1, 0.1);
        final NumberAxis yAxis = new NumberAxis(-1, 1, 0.1);
        final ScatterChart<Number,Number> sc = new
                ScatterChart<>(xAxis, yAxis);
        xAxis.setLabel("Age (years)");
        yAxis.setLabel("Returns to date");
        sc.setTitle("Investment Overview");



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

        XYChart.Series series1 = new XYChart.Series();
        series1.setName("Type 0");
        //series1.getNode().lookup(".chart-series-area-fill").setStyle("-fx-fill: rgba(255, 100, 0, 1.0);");
        //series1.getData().add(new XYChart.Data(4.2, 193.2));

        XYChart.Series series3 = new XYChart.Series();
        series1.setName("");

        XYChart.Series series2 = new XYChart.Series();
        series2.setName("Type 1");
        data.forEach(instance -> {
            switch ((int) instance.value(2)) {
                case 0:
                    series1.getData().add(new XYChart.Data<>(instance.value(0), instance.value(1)));
                    break;
                case 1:
                    series2.getData().add(new XYChart.Data<>(instance.value(0), instance.value(1)));
                    break;
            }
        });



        //series2.getData().add(new XYChart.Data(5.2, 229.2));


        sc.getData().addAll(series1, series3, series2);
        Scene scene  = new Scene(sc, 500, 400);
        stage.setScene(scene);
        stage.show();
    }

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

        float p = 0.8f;
        int length = (int) (data.size() * p);
        Instances train = new Instances(data, 0, length);
        Instances test = new Instances(data, length, data.size() - length);

        Classifier classifier = new IBk() {
            {
                setKNN(5);
            }
        };
        classifier.buildClassifier(train);

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

    public static void main(String[] args) {
        launch(args);
    }
}