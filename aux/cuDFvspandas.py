#!/usr/bin/env python3

import pandas as pd
import time

csv_dataset = "../datasets/healthcare_dataset.csv"

# Start time measurement
start_time = time.time()

# load data set into dataframe
df = pd.read_csv(csv_dataset)

print (df)
