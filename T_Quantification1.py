import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Your existing code

# Initialize lists to store the data for the DataFrame
index_list = []
x_list = []
y_list = []

# # # Add the first data point to the DataFrame
# index_list.append(0)
# x_list.append(x[0])
# y_list.append(y[0])

# Add data for each step position selected in last_step_positions
for step_pos in last_step_positions:
    index_list.append(step_pos)
    x_list.append(x[step_pos])
    y_list.append(y[step_pos])

# Add the last data point to the DataFrame
# index_list.append(len(x) - 1)
# x_list.append(x[-1])
# y_list.append(y[-1])

# Create a DataFrame from the lists
df = pd.DataFrame({'Index': index_list, 'X': x_list, 'Y': y_list})

# Display the DataFrame
print(df)

if n == 3:
    
    point_pairs = [(0, 1), (2, 3),(4,5)] 
elif n == 2:
    
    point_pairs = [(0, 1), (2, 3)]     
else:
    point_pairs = [(0, 1)]  # Define as many pairs as needed

# Initialize a list to store the differences
differences = []

# Calculate the differences for each pair of points
for pair in point_pairs:
    diff = df['Y'].iloc[pair[1]] - df['Y'].iloc[pair[0]]
    differences.append(diff)

point_pairs_x = [(0, 2)]  # Define as many pairs as needed

# Initialize a list to store the differences
lt_values = []
# n=3
# Calculate the differences for each pair of points
for pair in point_pairs_x:
    
    if n==2:
        diff_1 = df['X'].iloc[pair[0]] - min(x)
        diff_2 = df['X'].iloc[pair[1]] - min(x)
        lt_values.append(diff_1)
        lt_values.append(diff_2)
    elif n==3:
        diff_1 = df['X'].iloc[pair[0]] - min(x)
        diff_2 = df['X'].iloc[pair[1]] - min(x)
        diff_3 = df['X'].iloc[pair[1]] - min(x)
        lt_values.append(diff_1)
        lt_values.append(diff_2)
        lt_values.append(diff_3)
    else:
        diff_1 = df['X'].iloc[pair[0]] - min(x)
        lt_values.append(diff_1)
        
min_x = min(x)
ea_x_value = min_x +0.01


plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Original Signal')
plt.axvline(x=ea_x_value, color='green', linestyle='--', label='EA_x')
plt.scatter(df['X'], df['Y'], color='red', label=f'Last {n_last_steps} Steps')
plt.xlabel('x')
plt.ylabel('Signal')
plt.title('Identified Last Steps in the Signal')
plt.legend()
plt.grid(True)
plt.show()
 
# Create a new DataFrame with the differences
tf = pd.DataFrame({"Type":Cell_type,'Condition': C,'Cell number': Cell_number,'Interval':Timepoint,'Curve':t ,
                   'Tet_F': differences, 'Tet_Length': lt_values, 'endadhesion': ea_x_value,'min_adh':min_x})
tf["Bin"] = np.arange(len(tf)) // 1   
# Display the new DataFrame
print(tf)

