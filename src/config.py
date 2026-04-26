from pathlib import Path

# path directory
root_path = Path(__file__).resolve().parents[1]

data_path = root_path / 'data' / 'Depression_Severity_Levels_Dataset.csv'
output_dir = root_path / 'outputs'
model_dir = output_dir / 'models'
syn_dir = output_dir / 'synthetic'
report_dir = output_dir / 'report'

RANDOM_SEED = 67

# dataset split
test_size = 0.2
val_size = 0.5 # split test into smaller val and test sets

# TF-IDF baseline
TFIDF_max_features = 200000

# Classification models baseline
distilbert = 'distilbert-base-uncased'
mentalbert = 'mental/mental-bert-base-uncased'
max_len_transformer = 256
epochs = 2
LR = 2e-5
train_batch = 128
eval_batch = 256

# synthetic generators
synthetic_llm_falc = 'tiiuae/Falcon3-7B-Instruct'
synthetic_llm_ment = 'klyang/MentaLLaMA-chat-7B'
max_tokens = 200
batch_size = 32
gen_temp = 0.9
gen_top_p = 0.95
N_per_label = 10000

# temperature and k value to test
temp_values = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
k_value = 8