import os, torch, gc, subprocess
import pandas as pd
from config import (
    output_dir, syn_dir, temp_values,
    synthetic_llm_ment, synthetic_llm_falc
)
from src.synthetic_generation import pipeline_t, generate_posts
from src.prompts import build_prompt_list
from src.utils import save_df

def run_temperatures(train_path):
    df = pd.read_csv(train_path).dropna()
    labels = df['label'].unique()

    # build prompt lists
    print('Building prompt lists')
    prompts_zero, y_zero = build_prompt_list(labels, 'zero-shot', df)
    prompts_few, y_few = build_prompt_list(labels, 'few-shot', df)

    all_paths = {}

    for temp in temp_values:
        temp_str = str(temp).replace('.', '_')
        temp_dir = syn_dir / f'temp_{temp_str}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f'\nTemperature: {temp}')

        # MentaLLaMA
        pipe_ment = pipeline_t(synthetic_llm_ment, temp_t=temp)

        print(f'\nMentaLLaMA Zero-Shot (temp={temp})')
        gen_mz = generate_posts(pipe_ment, synthetic_llm_ment, y_zero, prompts_zero)
        path_mz = temp_dir / 'syn_llama_zero_shot.csv'
        save_df(gen_mz, path_mz)

        # free memory
        del gen_mz
        gc.collect()
        torch.cuda.empty_cache()

        print(f'\nMentaLLaMA Few-Shot (temp={temp})')
        gen_mf = generate_posts(pipe_ment, synthetic_llm_ment, y_few, prompts_few)
        path_mf = temp_dir / 'syn_llama_few_shot.csv'
        save_df(gen_mf, path_mf)

        # clean up MentaLLaMA
        print('Clearing MentaLLaMA model')
        del pipe_ment
        gc.collect()
        torch.cuda.empty_cache()
        subprocess.run(['rm', '-rf', os.path.expanduser('~/.cache/huggingface/hub')], check=True)

        # Falcon
        pipe_falc = pipeline_t(synthetic_llm_falc, temp_t=temp)

        print(f'\nFalcon Zero-Shot (temp={temp})')
        gen_fz = generate_posts(pipe_falc, synthetic_llm_falc, y_zero, prompts_zero)
        path_fz = temp_dir / 'syn_falcon_zero_shot.csv'
        save_df(gen_fz, path_fz)

        # free memory
        del gen_fz
        gc.collect()
        torch.cuda.empty_cache()

        print(f'\nFalcon Few-Shot (temp={temp})')
        gen_ff = generate_posts(pipe_falc, synthetic_llm_falc, y_few, prompts_few)
        path_ff = temp_dir / 'syn_falcon_few_shot.csv'
        save_df(gen_ff, path_ff)

        # clean up Falcon
        print('Clearing Falcon model')
        del pipe_falc
        gc.collect()
        torch.cuda.empty_cache()
        subprocess.run(['rm', '-rf', os.path.expanduser('~/.cache/huggingface/hub')], check=True)

        # save paths
        all_paths[temp] = {
            'llama_zero': path_mz,
            'llama_few':  path_mf,
            'falcon_zero': path_fz,
            'falcon_few':  path_ff,
        }

        print(f'\n  Finished temperature {temp}  ->  {temp_dir}')

    return all_paths

train_p = output_dir / 'processed_train.csv'

print(f'GPU status: {torch.cuda.is_available()}')
print(f'Temperatures to iterate through: {temp_values}\n')

results = run_temperatures(train_p)