import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

dataroot = Path("/home/viliar/Documents/VKMADE/contest/data")

df = pd.read_csv(dataroot / "train.csv")

classes = sorted(list(set(df["label"])))

train_parts = []
val_parts = []
for C in classes:
    part_data = df[df["label"] == C]

    train_part, val_part = train_test_split(
        part_data, test_size=0.1, random_state=42, shuffle=True
    )

    train_parts.append(train_part)
    val_parts.append(val_part)

train_result = pd.concat(train_parts)
val_result = pd.concat(val_parts)

train_result.reset_index(drop=True, inplace=True)
val_result.reset_index(drop=True, inplace=True)

train_result.to_csv(dataroot / "train_part.csv", index=False)
val_result.to_csv(dataroot / "val_part.csv", index=False)
