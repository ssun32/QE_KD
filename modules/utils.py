def convert_to_class(score, dataset="qe", normalized=True, binary=False):
    if normalized:
        score *= 100
        score = max(min(100, score), 0)

    if dataset == "qe":
        if 0 <= score <= 10:
            da_class = 0
        elif 10 < score <= 29:
            da_class = 1
        elif 29 < score <= 50:
            da_class = 2
        elif 50 < score <= 69:
            da_class = 3
        elif 69 < score <= 90:
            da_class = 4
        elif 90 < score <= 100:
            da_class = 5

        if binary:
            da_class = int(da_class >= 3)
    else:
        if 0 <= score < 20:
            da_class = 0
        elif 20 <= score < 30:
            da_class = 1
        elif 30 <= score <= 100:
            da_class = 2

        if binary:
            da_class = int(da_class == 1)
    return da_class

