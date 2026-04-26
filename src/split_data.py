import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, test_size, val_size
from utils import save_df

def split_save(df: pd.DataFrame, train_path, val_path, test_path):
    # stratified split to maintain class balance
    train_df, temp_df = train_test_split(
        df, test_size = test_size, random_state=RANDOM_SEED, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, random_state=RANDOM_SEED, stratify=temp_df['label']
    )

    save_df(train_df, train_path)
    save_df(val_df, val_path)
    save_df(test_df, test_path)
    
    return train_df, val_df, test_df