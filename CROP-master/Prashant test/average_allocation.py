import pandas as pd

# Read data from the first CSV file
df1 = pd.read_csv('/home/prashant/CROP-master/Prashant test/standard_allocation.csv')

# Read data from the second CSV file
df2 = pd.read_csv('/home/prashant/CROP-master/Prashant test/new_allocation.csv')

df1_numeric = df1.apply(pd.to_numeric, errors='coerce')
df2_numeric = df2.apply(pd.to_numeric, errors='coerce')

# Calculate the average of corresponding columns
average_df = (df1_numeric + df2_numeric) / 2
# Write the averages to a new CSV file
average_df.to_csv('/home/prashant/CROP-master/Prashant test/average_allocation.csv', index=False)

