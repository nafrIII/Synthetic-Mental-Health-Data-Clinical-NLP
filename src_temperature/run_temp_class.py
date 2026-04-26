from config import output_dir, syn_dir
from src.baseline_bert import run_baseline_classifiers
import torch, gc

val_p = output_dir / 'processed_val.csv'
test_p = output_dir / 'processed_test.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Free any leftover GPU memory before starting
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f'GPU status: {torch.cuda.is_available()}')

synthetic_datasets = {'syn_falcon_zero_shot.csv': 'syn_fal_ZS_', 
                      'syn_falcon_few_shot.csv': 'syn_fal_FS_', 
                      'syn_llama_zero_shot.csv': 'syn_ML_ZS_', 
                      'syn_llama_few_shot.csv': 'syn_ML_FS_'}

temp_values = [0.8, 1.0]

for x in temp_values:
    temp_str = str(x).replace('.', '_')
    temp_dir = syn_dir / f'temp_{temp_str}'
    for k, v in synthetic_datasets.items():
        print('running classification for:', k, 'at temp:', x)
        path = temp_dir / k
        path_name = v + temp_str

        temp_results = run_baseline_classifiers(path, val_p, test_p, path_name)
