import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/prashant/CROP-master/Prashant test/2021_wheat.csv')

# Print unique values from the 'district_name' column
unique_districts = df['District Name'].unique()
print("Unique districts:")
for district in unique_districts:
    print(district)
average_prices_min = df.groupby('District Name')['Min Price (Rs./Quintal)'].mean()
average_prices_max = df.groupby('District Name')['Max Price (Rs./Quintal)'].mean()
average_prices_mod = df.groupby('District Name')['Modal Price (Rs./Quintal)'].mean()

# Print the average prices for each district
print("Average prices for each district:")
print(average_prices_min,average_prices_max,average_prices_mod)

average_data = pd.DataFrame({'Average_Price_min': average_prices_min, 'Average_prices_max': average_prices_max, 'Average_prices_modal':average_prices_mod})

# Save the DataFrame to a CSV file
output_path = '/home/prashant/CROP-master/Prashant test/wheat_price_2021.csv'
average_data.to_csv(output_path) 
