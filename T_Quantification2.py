
import numpy as np
import matplotlib.pyplot as plt
#### find the points: 
arr = []
ssc=1000
y =  c_f_def[0:ssc]
x =c_f_ts[0:ssc]

# Assuming 'x' and5'y' are your original signal data

# Reverse the order of the signal
x_reversed = np.flip(x)
y_reversed = np.flip(y)

# Calculate the standard deviation of the last 100 points of the reversed signal
rolling_std = 0.28#np.td(y_reversed[:100])

# Initialize a list to store the positions of detected steps and the points before them
step_positions = []

# Loop through the reversed signal to detect steps
for i in range(2, len(y_reversed)):
    if abs(y_reversed[i] - y_reversed[i-2]) > rolling_std:
        step_positions.append(len(y_reversed) - i+12)  # Step position
        step_positions.append(len(y_reversed) - i -1)  # Point before the step

# Reverse the step positions to match the original order
step_positions.reverse()
n=1
# Specify the number of steps to select
n_last_steps = 2*n

# Select only the last two steps from the ones identified in the loop
last_step_positions = step_positions[-n_last_steps:]

# Extract the magnitudes of the steps
# step_magnitudes = np.abs(y[last_step_positions[0]] - y[last_step_positions[1]])

# Plot the original signal and identified steps
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Original Signal')
plt.scatter(x[last_step_positions], y[last_step_positions],s = 200,color='red', label=f'Last {n_last_steps} Steps')
plt.xlabel('x')
plt.ylabel('Signal')
plt.title('Identified Last Steps in the Signal')
plt.legend()
plt.grid(True)
plt.show()

# Output the positions and magnitudes of the last two identified steps
print("Last", n_last_steps, "Step Positions:", x[last_step_positions])
# print("Last", n_last_steps, "Step Size:", step_magnitudes)

arr.append( [Timepoint, Cell_number, n, min(y)])

