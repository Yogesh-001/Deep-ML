"""Linear Regression - Power Grid Optimization
Medium
Machine Learning


It is the year 2157. Mars has its first thriving colony, and energy consumption is steadily on the rise. As the lead data scientist, you have daily power usage measurements (10 days) affected by both a growing linear trend and a daily fluctuation. The fluctuation follows the formula fáµ¢ = 10 x sin(2Ï x i / 10), where i is the day number (1 through 10). Your challenge is to remove this known fluctuation from each data point, fit a linear regression model to the detrended data, predict day 15's base consumption, add back the fluctuation for day 15, and finally include a 5% safety margin. The final answer must be an integer, ensuring you meet the colony's future needs.

Example:
Input:
Daily consumption data for 10 days: [150, 165, 185, 195, 210, 225, 240, 260, 275, 290]
Output:
404
Reasoning:
For each of the 10 days, we subtract the daily fluctuation given by 10xsin(2πxi/10). We then perform linear regression on the resulting values, predict day 15’s base usage, and add back the day 15 fluctuation. Finally, we apply a 5% margin. Running the provided solution code yields 404 for this dataset."""

import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	fluctuations = [consumption_data[i - 1] - (10 * math.sin(2 * PI * i / 10)) for i in range(1, n+1)]
	# sum_x = sum(range(1,n+1))
	# sum_x2 = sum(i ** 2 for i in range(1,n+1))
	sum_x = (n * (n + 1)) // 2
  sum_x2 = (n * (n + 1) * (2 * n + 1)) // 6
	sum_y = sum(fluctuations)
	sum_xy = sum(i * fluctuations[i-1] for i in range(1, n+1))

	m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) #slope
	b = (sum_y - m * sum_x) / n #intercept

	# y = mx + b
	base_15 = m * 15 + b
	pred_15 = base_15 + (10 * math.sin(2 * PI * 15) / 10)

	return math.ceil(pred_15 * 1.05)
