import torch
import pandas as pd
from config import data_path, output_dir
from preprocessing import preprocess_df
from split_data import split_save
from synthetic_generation import run_generation
from baseline_bert import run_baseline_classifiers

train_p = output_dir / 'processed_train.csv'
val_p = output_dir / 'processed_val.csv'
test_p = output_dir / 'processed_test.csv'

df = pd.read_csv(data_path)

df = preprocess_df(df)

split_save(df, train_p, val_p, test_p)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'GPU status: {torch.cuda.is_available()}')

# Run baseline benchmark
baseline_results = run_baseline_classifiers(train_p, val_p, test_p, 'baseline')