Report: Analysis of Hubble’s Law and Estimation of the Universe’s Age
1. Data Description
The dataset includes observational measurements of galaxies' distances (dd) and velocities (vv), capturing the relationship described by Hubble's Law (v=H⋅d):
•	Distance (d): Measured in megaparsecs (Mpc), indicating galaxies' distances from Earth.
•	Velocity (v): Measured in kilometers per second (km/s), representing how fast galaxies are moving away due to redshift.
Summary:
•	Data points: 19 galaxy observations.
•	The dataset captures the relationship described by Hubble's Law (v=H⋅dv = H \cdot d), with HH being the Hubble constant.

3. Objectives of the Analysis
•	Primary Goal: Estimate the Hubble constant (H) using a linear regression model applied to the observed dd-vv relationship.
•	Secondary Goal: Use the estimated H to calculate the age of the universe assuming uniform expansion.
•	Deliverable: Statistical evaluation and confidence intervals to support insights into the universe's expansion.
•	Provide insights into the universe's expansion through statistical evaluation and confidence intervals.

5. Model Selection
Several modeling approaches were considered:
1.	Linear Regression:
o	Using the least-squares method to fit v against d, ensuring the line passes through the origin (0,0).
o	Tools: SciPy's curve_fit() and Scikit-learn's Linear Regression model.
2.	Confidence Intervals:
o	Bootstrap resampling was applied to assess the uncertainty in model parameters (slope and intercept).
Model Performance:
•	Based on R^2 (Coefficient of Determination), the linear regression model explained approximately 88% of the variation in velocity as a function of distance.

4. Key Findings
1.	Hubble Constant (H):
o	Estimated H=60.35H = 60.35 km/s/Mpc (95% confidence interval provided).
o	Represents the rate of universe expansion.
2.	Age of the Universe (t):
o	Estimated using t=1/Ht = 1/H: ~16.58 billion years (with uncertainty included).
o	Highlights an approximation dependent on the assumption of uniform expansion.
o	Assumes uniform expansion over time.
3.	Statistical Metrics:
o	Residual Sum of Squares (RSS): 22,006,329.
o	Mean Squared Error (MSE): 1,158,228.
o	Degree of Freedom (DOF): 17.
o	Model Coefficients: Slope (HH): 60.35, Intercept: 494.41.
o	R2=0.88R^2 = 0.88.

5. Model Limitations and Recommendations
•	Flaws:
o	The intercept deviates from zero, which may indicate noise or systematic errors in measurements.
o	Assumption of uniform expansion ignores potential time-dependent changes in Hubble constant.
•	Future Steps:
o	Collect additional data points for galaxies at greater distances.
o	Explore more advanced regression models, such as weighted least squares, to handle variance in observational uncertainty.
o	Consider non-linear expansions of Hubble's Law to account for variations in the universe's expansion rate.
Recommendations for Future Analysis:
•	Data Collection: Gather additional data for galaxies at greater distances to refine estimates.
•	Model Improvements:
o	Explore advanced regression models, such as weighted least squares, to account for variance in observational uncertainty.
o	Investigate non-linear expansions of Hubble's Law to capture variations in the universe's expansion rate.
•	Interdisciplinary Collaboration: Engage with astrophysics experts to incorporate cosmological models that accommodate time-variable HH.

6. Appendix (Optional)
•	Python code for:
o	Plotting the d-v relationship.
o	Implementing linear regression with confidence intervals.
o	Calculating metrics like RSS, MSE, and R^2.
7.My pythoncode
Plot the Relationship Between d and v
import matplotlib.pyplot as plt
# Data from the provided table
distance = [15, 97, 32, 145, 50, 122, 58, 91, 120, 93, 158, 64, 145, 61, 103, 46, 34, 185, 20]
velocity = [1100, 6700, 2400, 10700, 3100, 9900, 4300, 5300, 9000, 7500, 8900, 5300, 9600, 3300, 5100, 3600, 1800, 9500, 1200]
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(distance, velocity, color='blue', label='Data Points')
plt.xlabel('Distance (Mpc)', fontsize=12)
plt.ylabel('Velocity (km/s)', fontsize=12)
plt.title('Relationship Between Distance and Velocity (Hubble\'s Law)', fontsize=14)
plt.legend()
plt.grid()
plt.show()
Output:
 
Analysis: The image displayed provides a graphical representation of Hubble's Law. It demonstrates the relationship between the distance of galaxies and their velocity, where the data points clearly show a positive correlation: the farther a galaxy is, the faster it moves away. This insight is critical for astronomers, as it supports the understanding of the expanding universe and provides a foundation for estimating the Hubble constant and, subsequently, the age of the universe.’
Classifies galaxies as either "fast-moving" or "slow-moving" based on their velocity.
import pandas as pd

# Data: Galaxy velocity
data = {
    "distance": [15, 97, 32, 145, 50, 122, 58, 91, 120, 93, 158, 64, 145, 61, 103, 46, 34, 185, 20],
    "velocity": [1100, 6700, 2400, 10700, 3100, 9900, 4300, 5300, 9000, 7500, 8900, 5300, 9600, 3300, 5100, 3600, 1800, 9500, 1200]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define a velocity threshold for categorization (e.g., 5000 km/s)
velocity_threshold = 5000

# Create a new categorical outcome variable
df["velocity_category"] = ["fast-moving" if v > velocity_threshold else "slow-moving" for v in df["velocity"]]

# Display the updated DataFrame
print(df)
Output:
  distance  velocity velocity_category
0         15      1100       slow-moving
1         97      6700       fast-moving
2         32      2400       slow-moving
3        145     10700       fast-moving
4         50      3100       slow-moving
5        122      9900       fast-moving
6         58      4300       slow-moving
7         91      5300       fast-moving
8        120      9000       fast-moving
9         93      7500       fast-moving
10       158      8900       fast-moving
11        64      5300       fast-moving
12       145      9600       fast-moving
13        61      3300       slow-moving
14       103      5100       fast-moving
15        46      3600       slow-moving
16        34      1800       slow-moving
17       185      9500       fast-moving
18        20      1200       slow-moving
Explanation
1.	Threshold: The threshold (5000 km/s) separates galaxies into "fast-moving" (v>5000v > 5000) and "slow-moving" (v≤5000v \leq 5000).

Plot the Predicted Line with 95% Confidence Interval
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Prepare the data
distance_np = np.array(distance).reshape(-1, 1)
velocity_np = np.array(velocity)

# Fit linear regression model
model = LinearRegression()
model.fit(distance_np, velocity_np)

# Predicted values
predicted_velocity = model.predict(distance_np)

# Calculate confidence intervals (bootstrap method)
bootstrap_slopes = []
bootstrap_intercepts = []

for _ in range(1000):
    resampled_distance, resampled_velocity = resample(distance_np, velocity_np)
    bootstrap_model = LinearRegression()
    bootstrap_model.fit(resampled_distance, resampled_velocity)
    bootstrap_slopes.append(bootstrap_model.coef_[0])
    bootstrap_intercepts.append(bootstrap_model.intercept_)

slope_confidence = np.percentile(bootstrap_slopes, [2.5, 97.5])
intercept_confidence = np.percentile(bootstrap_intercepts, [2.5, 97.5])

# Plot data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(distance, velocity, color='blue', label='Data Points')
plt.plot(distance, predicted_velocity, color='red', label='Regression Line')

# Plot confidence interval
lower_bound = slope_confidence[0] * np.array(distance) + intercept_confidence[0]
upper_bound = slope_confidence[1] * np.array(distance) + intercept_confidence[1]
plt.fill_between(distance, lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Confidence Interval')

plt.xlabel('Distance (Mpc)', fontsize=12)
plt.ylabel('Velocity (km/s)', fontsize=12)
plt.title('Linear Regression with 95% Confidence Interval', fontsize=14)
plt.legend()
plt.grid()
plt.show()
Output:




 
Analysis: The graph visually represents Hubble's Law, plotting the relationship between the distance of galaxies ($d$) and their velocity ($v$), with observational data and a fitted linear regression line. The red line through the data passes through the origin (0,0), reflecting the proportional relationship dictated by the equation v=Hdv = Hd. This is essential for deriving the Hubble constant and analyzing the expansion of the universe.

from sklearn.metrics import mean_squared_error, r2_score

# Residuals
residuals = velocity_np - predicted_velocity

# Residual Sum of Squares (RSS)
RSS = np.sum(residuals**2)

# Degree of Freedom (DOF)
DOF = len(velocity_np) - 2  # Number of data points - number of parameters (slope + intercept)

# Mean Squared Error (MSE)
MSE = mean_squared_error(velocity_np, predicted_velocity)

# Coefficient of Determination (R^2)
R2 = r2_score(velocity_np, predicted_velocity)

# Model Coefficients
slope = model.coef_[0]
intercept = model.intercept_

# Display
print(f"Residual Sum of Squares (RSS): {RSS}")
print(f"Degree of Freedom (DOF): {DOF}")
print(f"Model Coefficients: Slope = {slope}, Intercept = {intercept}")
print(f"Mean Squared Error (MSE): {MSE}")
print(f"Coefficient of Determination (R^2): {R2}")
Output:
Residual Sum of Squares (RSS): 22006329.225289907
Degree of Freedom (DOF): 17
Model Coefficients: Slope = 60.345485509960376, Intercept = 494.4078552197343
Mean Squared Error (MSE): 1158227.8539626268
Coefficient of Determination (R^2): 0.8808794563966119
Analysis:
  Residual Sum of Squares (RSS):
•	Value: 22,006,329.23
•	This indicates the total variation in the observed data that remains unexplained by the model. A lower RSS suggests a better fit, and this value shows that the model does reasonably well.
  Degree of Freedom (DOF):
•	Value: 17
•	This is calculated as the number of data points minus the number of parameters in the model (slope and intercept). It helps assess the variability of the data.
 Model Coefficients:
•	Slope (Hubble Constant): 60.35 km/s/Mpc
•	Intercept: 494.41 km/s
•	The slope represents the Hubble constant HH, which indicates the rate of expansion of the universe. The intercept is the velocity at zero distance, which ideally should be close to zero but is slightly offset here, likely due to noise or measurement errors.
Mean Squared Error (MSE):
•	Value: 1,158,227.85
•	This represents the average squared difference between predicted and actual values. Lower MSE indicates better predictive performance.
  Coefficient of Determination (R2R^2):
•	Value: 0.88
•	This means that approximately 88% of the variation in the velocity (vv) is explained by the distance (dd) in this model, which reflects a strong linear relationship

