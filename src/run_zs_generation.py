from config import output_dir, syn_dir
from synthetic_generation import run_generation_zs_ment, run_generation_zs_falc
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
# zero-shot
ml_zs_p = run_generation_zs_ment(train_p)
ml_zs_p = ml_zs_p['llama_zero']
ml_zs_results = run_baseline_classifiers(ml_zs_p, val_p, test_p, 'synthetic_ML_ZS')

# falcon
# zero-shot
f_zs_p = run_generation_zs_falc(train_p)
f_zs_p = f_zs_p['falcon_zero']
f_zs_results = run_baseline_classifiers(f_zs_p, val_p, test_p, 'synthetic_fal_ZS')