"""Calculate P50/P95/P99 Latency Percentiles
Easy
Statistics


Implement a function that calculates the P50, P95, and P99 latency percentiles from a list of latency measurements. These percentiles are critical metrics for monitoring system performance, API response times, and model inference latencies in production ML systems.

The function should take a list of latency values (in milliseconds or any time unit) and return a dictionary containing the 50th, 95th, and 99th percentile values, each rounded to 4 decimal places.

Specifications:

P50 (median): The value below which 50% of observations fall
P95: The value below which 95% of observations fall
P99: The value below which 99% of observations fall
Use linear interpolation for percentile calculation
If the input list is empty, return all percentiles as 0.0
Return values should be rounded to 4 decimal places
Example:
Input:
latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Output:
{'P50': 55.0, 'P95': 95.5, 'P99': 99.1}
Reasoning:
With 10 sorted values, P50 falls between indices 4 and 5 (values 50 and 60), interpolating to 55.0. P95 position is 8.55, interpolating between 90 and 100 gives 95.5. P99 position is 8.91, interpolating between 90 and 100 gives 99.1. These percentiles show that 50% of requests complete within 55ms, 95% within 95.5ms, and 99% within 99.1ms."""

import numpy as np

def latency_index(latencies, P):
    return (len(latencies) - 1) * P / 100

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here
    if not latencies:
        return {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
    
    sorted_latencies = sorted(latencies)
    percentiles = {}

    for label, pos in [('P50' , 50), ('P95' , 95), ('P99' , 99)]:
        index = latency_index(sorted_latencies, pos)

        idx_low = int(index)
        idx_high = idx_low + 1

        if idx_high < len(latencies):
            frac = index - idx_low
            val = sorted_latencies[idx_low] + frac * (sorted_latencies[idx_high] - sorted_latencies[idx_low])
        
        else:
            val = sorted_latencies[idx_low]
        
        percentiles[label] = round(float(val), 4)

    return percentiles
