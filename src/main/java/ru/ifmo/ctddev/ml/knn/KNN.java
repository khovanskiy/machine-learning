package ru.ifmo.ctddev.ml.knn;

import javafx.util.Pair;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * @author victor
 */
public class KNN extends AbstractClassifier {

    protected int k = 12;
    protected PriorityQueue<Pair<Double, Instance>> queue = new PriorityQueue<>();
    protected Instances data;

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.data = data;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        PriorityQueue<Pair<Double, Instance>> queue = new PriorityQueue<>((l, r) -> Double.compare(l.getKey(), r.getKey()));

        for (Instance other : data) {
            assert other != null;
            assert instance.numAttributes() == other.numAttributes();
            double total = 0.0;
            for (int i = 0; i < other.numAttributes(); ++i) {
                if (other.classIndex() == i) {
                    continue;
                }
                assert instance.attribute(i).type() == other.attribute(i).type();
                switch (instance.attribute(i).type()) {
                    case Attribute.NUMERIC:
                        double a = instance.value(i);
                        double b = other.value(i);
                        double c = Math.abs(a - b);
                        total += c * c;
                }
            }
            queue.add(new Pair<>(Math.sqrt(total), other));
        }

        int ava = Math.min(k + 1, queue.size());
        List<Pair<Double, Instance>> instances = new ArrayList<>();
        for (int i = 0; i < ava; i++) {
            instances.add(queue.poll());
        }
        double h = instances.get(ava - 1).getKey();

        Map<Integer, Double> weights = new HashMap<>();
        for (int i = 0; i < ava - 1; ++i) {
            Pair<Double, Instance> p = instances.get(i);
            Instance o = p.getValue();
            int x = (int) o.value(o.classIndex());
            double a = weights.getOrDefault(x, 0d);
            double dist = p.getKey() / h;
            double b = a + 0.75 * (1 - dist * dist);
            weights.put(x, b);
        }
        double max = -1;
        double cc = Utils.missingValue();
        for (Map.Entry<Integer, Double> entry : weights.entrySet()) {
            if (max < entry.getValue()) {
                max = entry.getValue();
                cc = entry.getKey();
            }
        }
        assert cc == 0 || cc == 1;
        //System.out.println(cc);
        return cc;
    }
}
