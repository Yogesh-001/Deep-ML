"""Implement Prediction Distribution Monitoring
Medium
MLOps


Implement a function to monitor changes in model prediction distributions between a reference (baseline) period and a current period. This is a critical MLOps task for detecting model drift in production.

Given two lists of prediction scores (probabilities between 0 and 1), compute the following monitoring metrics:

Mean Shift: The difference between the mean of current predictions and the mean of reference predictions
Standard Deviation Ratio: The ratio of current standard deviation to reference standard deviation
Jensen-Shannon Divergence: A symmetric measure of distribution similarity based on histogram comparison
Drift Detected: A boolean flag indicating if JS divergence exceeds 0.1 (significant drift threshold)
For the Jensen-Shannon divergence calculation:

Create histograms using n_bins equally spaced bins between 0 and 1
Apply Laplace smoothing to handle empty bins: P(bin) = (count + 1) / (total + n_bins)
Compute JS divergence as the average of KL divergences from each distribution to their mixture
Return a dictionary with keys 'mean_shift', 'std_ratio', 'js_divergence', and 'drift_detected'.

Example:
Input:
reference_preds = [0.1, 0.2, 0.3, 0.4, 0.5]
current_preds = [0.5, 0.6, 0.7, 0.8, 0.9]
n_bins = 5
Output:
{'mean_shift': 0.4, 'std_ratio': 1.0, 'js_divergence': 0.2939, 'drift_detected': True}
Reasoning:
The reference predictions have mean 0.3 and the current predictions have mean 0.7, giving a mean_shift of 0.4. Both have the same spread (std = 0.1414), so std_ratio = 1.0. The JS divergence of 0.2939 exceeds the 0.1 threshold, indicating significant distribution drift.
"""



import numpy as np

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    # Your code here
    curr_preds = np.array(current_preds)
    refer_preds = np.array(reference_preds)
    mean_shift = np.mean(curr_preds - refer_preds)

    std_ratio = np.std(curr_preds) / np.std(refer_preds)

    curr_counts, _ = np.histogram(curr_preds, bins = n_bins, range = (0,1))
    refer_counts, _ = np.histogram(refer_preds, bins = n_bins, range = (0,1))

    P = (refer_counts + 1) / (len(refer_preds) + n_bins)
    Q = (curr_counts + 1) / (len(curr_preds) + n_bins)

    M = 0.5 * (P + Q)

    kl_pm = np.sum(P * np.log(P / M))
    kl_qm = np.sum(Q * np.log(Q / M))

    JS_divergence = 0.5 * kl_pm + 0.5 * kl_qm

    drift_detected = bool(JS_divergence > 0.1)

    return {
        'mean_shift' : mean_shift,
        'std_ratio' : std_ratio,
        'js_divergence' : JS_divergence,
        'drift_detected' : drift_detected
    }

