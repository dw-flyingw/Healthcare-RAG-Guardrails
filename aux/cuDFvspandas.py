#!/usr/bin/env python3

import pandas as pd
import cudf
import time

csv_dataset = "../datasets/healthcare_dataset.csv"

'''# Start time measurement
start_time = time.time()

# load data set into dataframe
df = pd.read_csv(csv_dataset)

print (df)

import pandas as pd
import cudf
import time

# Create a Pandas DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Convert the Pandas DataFrame to a cuDF DataFrame
gdf = cudf.DataFrame.from_pandas(df)

# Measure the time it takes to perform a calculation on the Pandas DataFrame
start_time = time.time()
df['C'] = df['A'] + df['B']
end_time = time.time()
pandas_time = end_time - start_time

# Measure the time it takes to perform the same calculation on the cuDF DataFrame
start_time = time.time()
gdf['C'] = gdf['A'] + gdf['B']
end_time = time.time()
cudf_time = end_time - start_time

# Print the results
print("Pandas time:", pandas_time)
print("cuDF time:", cudf_time)

'''