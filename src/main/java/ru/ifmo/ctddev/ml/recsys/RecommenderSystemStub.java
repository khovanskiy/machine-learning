package ru.ifmo.ctddev.ml.recsys;

import lombok.extern.slf4j.Slf4j;
import ru.ifmo.ctddev.ml.util.CollectionUtils;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
@Slf4j
public class RecommenderSystemStub implements RecommenderSystem {
    private final Map<Long, Map<Long, Rating>> userItemRatings = new HashMap<>();
    private final int k;
    private double averageRating;

    public RecommenderSystemStub(int k) {
        this.k = k;
    }

    @Override
    public Rating predict(long userId, long itemId) {
        PriorityQueue<Weight> queue = new PriorityQueue<>(Comparator.comparingDouble(Weight::getValue).reversed());
        //log.info("Users' similarity is calculating...");
        if (!userItemRatings.containsKey(userId)) {
            return new Rating(userId, itemId, averageRating);
        }
        for (Map.Entry<Long, Map<Long, Rating>> entry : userItemRatings.entrySet()) {
            long neighbourUserId = entry.getKey();
            if (userId == neighbourUserId) {
                continue;
            }
            Rating r = getRating(neighbourUserId, itemId);
            if (r == null) {
                continue;
            }
            double weight = similarity(userId, neighbourUserId);
            queue.add(new Weight(neighbourUserId, weight));
        }
        List<Weight> nearest = new ArrayList<>(k);
        while (!queue.isEmpty() && nearest.size() < 5) {
            Weight weight = queue.poll();
            nearest.add(weight);
        }
        double sum1 = 0;
        for (Weight weight : nearest) {
            double w = weight.getValue();
            Rating r = getRating(weight.getUserId(), itemId);
            double vi = r.getValue();
            sum1 += w * vi;
        }
        double sum2 = 0;
        for (Weight weight : nearest) {
            sum2 += Math.abs(weight.getValue());
        }
        double predicted = sum1 / sum2;
        return new Rating(userId, itemId, predicted);
    }

    private static class Weight {
        private final long userId;
        private final double value;

        private Weight(long userId, double value) {
            this.userId = userId;
            this.value = value;
        }

        public long getUserId() {
            return userId;
        }

        public double getValue() {
            return value;
        }
    }

    @Override
    public void learn(Collection<? extends Rating> ratings) {
        for (Rating rating : ratings) {
            setRating(rating);
        }
        averageRating = ratings.stream().mapToDouble(Rating::getValue).average().orElse(0);
    }

    private void setRating(Rating rating) {
        Map<Long, Rating> set = userItemRatings.computeIfAbsent(rating.getUserId(), k -> new HashMap<>());
        set.put(rating.getItemId(), rating);
    }

    private Rating getRating(long userId, long itemId) {
        return userItemRatings.getOrDefault(userId, Collections.emptyMap()).get(itemId);
    }

    private double similarity(long userId1, long userId2) {
        Collection<Rating> set1 = userItemRatings.getOrDefault(userId1, Collections.emptyMap()).values();
        Collection<Rating> set2 = userItemRatings.getOrDefault(userId2, Collections.emptyMap()).values();
        List<Long> commonItems = CollectionUtils.intersect(set1, set2, Comparator.comparingLong(Rating::getItemId)).stream()
                .map(Rating::getItemId)
                .collect(Collectors.toList());
        if (commonItems.isEmpty()) {
            return 0;
        }
        double sumCommon = 0;
        for (Long itemId : commonItems) {
            sumCommon += getRating(userId1, itemId).getValue() * getRating(userId2, itemId).getValue();
        }
        double sum1 = 0;
        for (Rating rating : set1) {
            sum1 += rating.getValue() * rating.getValue();
        }
        double sum2 = 0;
        for (Rating rating : set2) {
            sum2 += rating.getValue() * rating.getValue();
        }
        double cos = sumCommon / Math.sqrt(sum1 * sum2);
        return cos;
    }
}
