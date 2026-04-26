import torch, gc
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import synthetic_llm_falc, synthetic_llm_ment, max_tokens, gen_temp, gen_top_p, N_per_label, syn_dir, batch_size
from prompts import build_prompt_list
from utils import save_df

def pipeline_t(model_id, temp_t=gen_temp):
  print(f'Model selected: {model_id}; temperature={temp_t}')
  tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    use_fast=False
  )

  tokenizer.padding_side = 'left'
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  
  if model_id == synthetic_llm_falc:
    model = AutoModelForCausalLM.from_pretrained(
      model_id,
      device_map='auto',
      dtype=torch.bfloat16,
      trust_remote_code=False,
      attn_implementation='sdpa'
    )
  else:
    model = AutoModelForCausalLM.from_pretrained(
      model_id,
      device_map='auto',
      dtype=torch.bfloat16,
      trust_remote_code=True,
      attn_implementation='sdpa'
    )
  
  pipe = pipeline(
    'text-generation', 
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=max_tokens,
    max_length=None,
    do_sample=True,
    temperature=temp_t,
    top_p=gen_top_p,
    truncation=True,
    batch_size=batch_size
  )
  return pipe


def generate_posts(pipe, model_id, labels, prompt_list, batch_size=batch_size):
  # prompt formatting
  if 'MentaLLaMA' in model_id:
    fmt = lambda p: f'<s>[INST] {p} [/INST]'
  elif 'falcon' in model_id:
    fmt = lambda p: f'User: {p}\nAssistant:'
  else:
    fmt = lambda p: p

  results = []
  formatted_prompts = [fmt(p) for p in prompt_list]
  print(f'\nGeneration for each label: {N_per_label}\nTotal generation: {len(prompt_list)}')

  for out, lab in tqdm(zip(pipe(formatted_prompts, batch_size=batch_size, return_full_text=False), labels), total=len(prompt_list)):
    raw_text = out[0]['generated_text']
    
    # try to extract content between <post> tags using regex
    post_match = re.search(r'<post>(.*?)(?:</post>|$)', raw_text, re.DOTALL)
    if post_match:
      cleaned = post_match.group(1)
    else:
      cleaned = raw_text

    # strip residual prompt-style formatting
    cleaned = cleaned.replace('User:', '').replace('Assistant:', '')
    cleaned = cleaned.replace('[INST]', '').replace('[/INST]', '')
    cleaned = cleaned.strip().strip('"').strip()

    if len(cleaned.split()) < 3:
      continue  # skip this row as it is too short to be a valid post
    
    if not re.search(r'[.!?\"\u2018\u2019*]+$', cleaned):
      cleaned += '.'

    results.append({'text': cleaned, 'label': lab})
  
  return pd.DataFrame(results)

def run_generation(train_path):
  df = pd.read_csv(train_path).dropna()
  labels = df['label'].unique()

  paths = {}

  # build prompt list
  print('Building prompt list')
  
  # zeroshot
  prompts_zero, y_zero = build_prompt_list(labels, 'zero-shot', df)
  # fewshot
  prompts_few, y_few = build_prompt_list(labels, 'few-shot', df)

  # MentaLLaMa
  pipe_ment = pipeline_t(synthetic_llm_ment)
  print('\nMentaLLaMa (Zero-Shot)\n')
  gen_mz = generate_posts(pipe_ment, synthetic_llm_ment, y_zero, prompts_zero)
  paths['llama_zero'] = syn_dir / 'syn_llama_zero_shot.csv'
  save_df(gen_mz, paths['llama_zero'])

  print('\nMentaLLaMa (Few-Shot)\n')
  gen_mf = generate_posts(pipe_ment, synthetic_llm_ment, y_few, prompts_few)
  paths['llama_few'] = syn_dir / 'syn_llama_few_shot.csv'
  save_df(gen_mf, paths['llama_few'])

  # clean model
  del pipe_ment
  gc.collect()
  torch.cuda.empty_cache()

  # falcon
  pipe_falc = pipeline_t(synthetic_llm_falc)
  print('\nFalcon (Zero-Shot)\n')
  gen_fz = generate_posts(pipe_falc, synthetic_llm_falc, y_zero, prompts_zero)
  paths['falcon_zero'] = syn_dir / 'syn_falcon_zero_shot.csv'
  save_df(gen_fz, paths['falcon_zero'])

  print('\nFalcon (Few-Shot)\n')
  gen_ff = generate_posts(pipe_falc, synthetic_llm_falc, y_few, prompts_few)
  paths['falcon_few'] = syn_dir / 'syn_falcon_few_shot.csv'
  save_df(gen_ff, paths['falcon_few'])

  # clean model
  del pipe_falc
  gc.collect()
  torch.cuda.empty_cache()

  return paths

def run_generation_zs_ment(train_path):
  df = pd.read_csv(train_path).dropna()
  labels = df['label'].unique()

  paths = {}

  # build prompt list
  print('Building prompt list')
  
  # zeroshot
  prompts_zero, y_zero = build_prompt_list(labels, 'zero-shot', df)

  # MentaLLaMa
  pipe_ment = pipeline_t(synthetic_llm_ment)
  print('\nMentaLLaMa (Zero-Shot)\n')
  gen_mz = generate_posts(pipe_ment, synthetic_llm_ment, y_zero, prompts_zero)
  paths['llama_zero'] = syn_dir / 'syn_llama_zero_shot.csv'
  save_df(gen_mz, paths['llama_zero'])

  # clean model
  del pipe_ment
  gc.collect()
  torch.cuda.empty_cache()

  return paths

def run_generation_fs_ment(train_path):
  df = pd.read_csv(train_path).dropna()
  labels = df['label'].unique()

  paths = {}

  # build prompt list
  print('Building prompt list')
  
  # fewshot
  prompts_few, y_few = build_prompt_list(labels, 'few-shot', df)

  # MentaLLaMa — use smaller batch for longer few-shot prompts
  fs_batch = max(1, batch_size // 4)  # 8 instead of 32
  pipe_ment = pipeline_t(synthetic_llm_ment)
  print('\nMentaLLaMa (Few-Shot)\n')
  gen_mf = generate_posts(pipe_ment, synthetic_llm_ment, y_few, prompts_few, batch_size=fs_batch)
  paths['llama_few'] = syn_dir / 'syn_llama_few_shot.csv'
  save_df(gen_mf, paths['llama_few'])

  # clean model
  del pipe_ment
  gc.collect()
  torch.cuda.empty_cache()

  return paths

def run_generation_zs_falc(train_path):
  df = pd.read_csv(train_path).dropna()
  labels = df['label'].unique()

  paths = {}

  # build prompt list
  print('Building prompt list')
  
  # zeroshot
  prompts_zero, y_zero = build_prompt_list(labels, 'zero-shot', df)

  # falcon
  pipe_falc = pipeline_t(synthetic_llm_falc)
  print('\nFalcon (Zero-Shot)\n')
  gen_fz = generate_posts(pipe_falc, synthetic_llm_falc, y_zero, prompts_zero)
  paths['falcon_zero'] = syn_dir / 'syn_falcon_zero_shot.csv'
  save_df(gen_fz, paths['falcon_zero'])

  # clean model
  del pipe_falc
  gc.collect()
  torch.cuda.empty_cache()

  return paths

def run_generation_fs_falc(train_path):
  df = pd.read_csv(train_path).dropna()
  labels = df['label'].unique()

  paths = {}

  # build prompt list
  print('Building prompt list')
  
  # fewshot
  prompts_few, y_few = build_prompt_list(labels, 'few-shot', df)

  # falcon — use smaller batch for longer few-shot prompts
  fs_batch = max(1, batch_size // 4)  # 8 instead of 32
  pipe_falc = pipeline_t(synthetic_llm_falc)
  print('\nFalcon (Few-Shot)\n')
  gen_ff = generate_posts(pipe_falc, synthetic_llm_falc, y_few, prompts_few, batch_size=fs_batch)
  paths['falcon_few'] = syn_dir / 'syn_falcon_few_shot.csv'
  save_df(gen_ff, paths['falcon_few'])

  # clean model
  del pipe_falc
  gc.collect()
  torch.cuda.empty_cache()

  return paths