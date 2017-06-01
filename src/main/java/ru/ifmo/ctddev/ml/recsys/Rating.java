package ru.ifmo.ctddev.ml.recsys;

import java.util.Objects;

/**
 * @author Victor Khovanskiy
 * @since 1.0.0
 */
public class Rating {
    private final long userId;
    private final long itemId;
    private final double value;

    public Rating(long userId, long itemId, double value) {
        this.userId = userId;
        this.itemId = itemId;
        this.value = value;
    }

    public long getUserId() {
        return userId;
    }

    public long getItemId() {
        return itemId;
    }

    public double getValue() {
        return value;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(userId) + Objects.hashCode(itemId);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof Rating && Objects.equals(this.userId, ((Rating) obj).userId) && Objects.equals(this.itemId, ((Rating) obj).itemId);
    }
}
