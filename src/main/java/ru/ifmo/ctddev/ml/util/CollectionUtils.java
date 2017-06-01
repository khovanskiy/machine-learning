package ru.ifmo.ctddev.ml.util;

import java.util.*;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
public class CollectionUtils {
    public static <T> Set<T> intersect(Collection<T> a, Collection<T> b, Comparator<? super T> comparator) {
        TreeSet<T> intersection = new TreeSet<>(comparator);
        intersection.addAll(a);
        TreeSet<T> wrapper = new TreeSet<>(comparator);
        wrapper.addAll(b);
        intersection.retainAll(wrapper);
        return intersection;
    }
}
