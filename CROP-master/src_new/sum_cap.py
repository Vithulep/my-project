import pandas as pd
import numpy as np

def read_csv_to_matrix(file_path, columns):
    """
    Reads specified columns from a CSV file and converts them to a NumPy array.

    :param file_path: str, the path to the CSV file
    :param columns: list of str, the column names to read
    :return: NumPy array containing the specified columns
    """
    # Read only the specified columns from the CSV file
    df = pd.read_csv(file_path, usecols=columns)

    # Convert the DataFrame to a NumPy array
    return df.values

def array_to_csv(array, csv_file_path, column_names=None):
    """
    Converts a NumPy array to a CSV file.

    :param array: NumPy array to be converted
    :param csv_file_path: str, the path where the CSV file will be saved
    :param column_names: list of str, the column names for the CSV file (optional)
    """
    # Convert the NumPy array to a DataFrame
    df = pd.DataFrame(array, columns=column_names)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

def main():
    # Replace 'your_file.csv' with the path to your CSV file
    file_path = '/home/prashant/CROP-master/examples/10_crops/Warehouse_allocation.csv'

    # Specify the columns you want to read
    columns = ['WH_ID', 'Capacity']

    # Read the data and convert to a matrix
    matrix = read_csv_to_matrix(file_path, columns) 
    ans=np.zeros((31,2),int)
    num_rows = matrix.shape[0]
    for i in range(num_rows):
        a=matrix[i][0] 
        if ans[a][1]==0: 
            sum=0
            for j in range(num_rows): 
                if matrix[j][0]==a: 
                    sum=sum+matrix[j][1]

            ans[a][1]=sum
            ans[a][0]=a        


    # Print the matrix
    # print(matrix)
    print(ans)
    csv_file_path = '/home/prashant/CROP-master/examples/10_crops/output.csv'

    # Optionally, specify the column names
    column_names = ['DID', 'capacity']

    # Convert the array to a CSV file
    array_to_csv(ans, csv_file_path, column_names)

    print(f"Array has been saved to {csv_file_path}")

if __name__ == "__main__":
    main()
