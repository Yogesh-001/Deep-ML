import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    data = np.array(data)
    results = {}
    
    # Central Tendency
    results['mean'] = np.mean(data)
    results['median'] = np.median(data)
    
    # Custom Mode Logic
    counts = {}
    for value in data:
        counts[value] = counts.get(value, 0) + 1
    max_freq = max(counts.values())
    modes = [key for key, val in counts.items() if val == max_freq]
    results['mode'] = modes[0] if len(modes) == 1 else min(modes)

    # Dispersion
    results['variance'] = np.var(data)
    results['standard_deviation'] = np.std(data)
    
    # Percentiles & IQR
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    results['25th_percentile'] = q1
    results['50th_percentile'] = q2
    results['75th_percentile'] = q3
    results['interquartile_range'] = q3 - q1
    
    # Rounding to 4 decimal places for consistency
    return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in results.items()}
