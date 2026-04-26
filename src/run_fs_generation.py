from config import output_dir
from synthetic_generation import run_generation_fs_falc, run_generation_fs_ment
from baseline_bert import run_baseline_classifiers
import torch, gc

train_p = output_dir / 'processed_train.csv'
val_p = output_dir / 'processed_val.csv'
test_p = output_dir / 'processed_test.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Free any leftover GPU memory before starting
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f'GPU status: {torch.cuda.is_available()}')

# MentaLLaMa
# few-shot
ml_fs_p = run_generation_fs_ment(train_p)
ml_fs_p = ml_fs_p['llama_few']
ml_fs_results = run_baseline_classifiers(ml_fs_p, val_p, test_p, 'synthetic_ML_FS')

# falcon
# few-shot
f_fs_p = run_generation_fs_falc(train_p)
f_fs_p = f_fs_p['falcon_few']
f_fs_results = run_baseline_classifiers(f_fs_p, val_p, test_p, 'synthetic_fal_FS')