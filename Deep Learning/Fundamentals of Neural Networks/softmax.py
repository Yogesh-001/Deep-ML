import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)

    exp_scores = [math.exp(score - max_score) for score in scores]

    total_sum = sum(exp_scores)

    softmax_values = [score / total_sum for score in exp_scores]

    return softmax_values
