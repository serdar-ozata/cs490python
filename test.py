import numpy as np
import util
import pandas as pd

# Read the CSV file
df = pd.read_csv("~/Downloads/uhem_results.csv")

# add new column to the DataFrame
df["improvement"] = np.nan

spmm_types = ["expand", "reduce"]

# Iterate over unique dataset names
for dataset_name in df["dataset_name"].unique():
    for spmm_type in spmm_types:
        tp_df = df[(df["dataset_name"] == dataset_name) & (df["spmm_type"] == spmm_type) & (df["comm_type"] == "tp")][0]
        # get the op value
        op_df = df[(df["dataset_name"] == dataset_name) & (df["spmm_type"] == spmm_type) & (df["comm_type"] == "op")][0]
        # calculate the improvement
        improvement = (op_df - tp_df) / op_df
        # update only the tp row
        df.loc[(df["dataset_name"] == dataset_name) & (df["spmm_type"] == spmm_type) & (
                df["comm_type"] == "tp"), "improvement"] = improvement
        # for op row leave "-" in the improvement column
        df.loc[(df["dataset_name"] == dataset_name) & (df["spmm_type"] == spmm_type) & (
                df["comm_type"] == "op"), "improvement"] = "-"

# Save the DataFrame to a new CSV file
df.to_csv("~/Downloads/uhem_results_improved.csv", index=False)
