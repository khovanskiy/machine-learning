package ru.ifmo.ctddev.ml.recsys;

import java.util.Collection;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
public interface RecommenderSystem {

    Rating predict(long userId, long itemId);

    void learn(Collection<? extends Rating> ratings);
}
