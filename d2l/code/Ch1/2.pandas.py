import os
import pandas as pd
from icecream import ic
import torch

os.makedirs(os.path.join("..", "data"), exist_ok=True)
data_file = os.path.join("..", "data", "house_tiny.csv")
with open(data_file, "w") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")


data = pd.read_csv(data_file)
ic(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(
    inputs.mean(numeric_only=True)
)  # to set numeric_only=True to only fill the missing values in numeric columns
ic(inputs)

inputs = pd.get_dummies(
    inputs, dummy_na=True, dtype=float
)  # to set dytpe=float so that the data type of the columns is float not bool
ic(inputs)


X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
ic(X, y)
