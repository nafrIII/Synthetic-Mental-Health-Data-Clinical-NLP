from pathlib import Path
import pandas as pd
import json

# functions for saving and loading data
def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_json(dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as x:
        json.dump(dict, x, indent=2)