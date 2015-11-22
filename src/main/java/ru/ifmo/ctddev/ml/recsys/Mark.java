package ru.ifmo.ctddev.ml.recsys;

/**
 * @author victor
 */
public class Mark {
    private final long user;
    private final long item;
    private final long value;

    public Mark(long user, long item, long value) {
        this.user = user;
        this.item = item;
        this.value = value;
    }

    public long getUser() {
        return user;
    }

    public long getItem() {
        return item;
    }

    public long getValue() {
        return value;
    }
}
