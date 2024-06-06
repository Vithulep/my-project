import csv

def calculate_average(yields):
    return round(sum(yields) / len(yields), 2)

def main():
    input_filename = '/home/prashant/CROP-master/MH/Book1.csv'  # Replace 'data.csv' with the name of your input CSV file
    output_filename = '/home/prashant/CROP-master/MH/store.csv'
    with open(input_filename, mode='r') as input_file, open(output_filename, mode='w', newline='') as output_file:
        reader = csv.DictReader(input_file)
        writer = csv.writer(output_file)
        
        # Write headers for the output file
        writer.writerow(['Row', 'Average Yield'])
        
        for row in reader:
            yield1 = float(row['yield1'])
            yield2 = float(row['yield2'])
            yield3 = float(row['yield3'])
            average_yield = calculate_average([yield1, yield2, yield3])
            writer.writerow([reader.line_num - 1, average_yield])

if __name__ == "__main__":
    main()
