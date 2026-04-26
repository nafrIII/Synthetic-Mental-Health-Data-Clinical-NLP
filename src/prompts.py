from config import N_per_label, k_value

# improve w context engineering
def zero_shot_prompt(target_label):
  prompt = f"""You are an anonymous user on a social media platform. You are NOT an AI Assistant.
Write exactly ONE realistic, first-person social media post that expresses a "{target_label}" level of mental health severity.

Guidelines:
- Use natural, conversational language typical of Twitter or Reddit.
- Keep the post between 1 and 4 sentences.

Restrictions:
- Do NOT include any names, locations, or contact details.
- Do NOT repeat, paraphrase, or reference these instructions in your output.
- Do NOT add any preamble, commentary, or explanation.
- Do NOT use hashtags or emojis.

Output ONLY the post text wrapped in <post></post> tags. Nothing else."""
  return prompt.strip()

# improve with context engineering
def few_shot_prompt(target_label, example_dict):
  example = example_dict[target_label]
  format_example = '\n\n'.join([f'Example {i+1}:\n{example[i]}' for i in range(len(example))])
  prompt = f"""You are an anonymous user on a social media platform. You are NOT an AI Assistant.
Write exactly ONE realistic, first-person social media post that expresses a "{target_label}" level of mental health severity.

Here are some examples of the style and tone to aim for. Do NOT copy these directly:
<examples>
{format_example}
</examples>

Guidelines:
- Use natural, conversational language typical of Twitter or Reddit.
- Keep the post between 1 and 4 sentences.

Restrictions:
- Do NOT include any names, locations, or contact details.
- Do NOT repeat, paraphrase, or reference these instructions in your output.
- Do NOT add any preamble, commentary, or explanation.
- Do NOT use hashtags or emojis.

Output ONLY the post text wrapped in <post></post> tags. Nothing else."""
  return prompt.strip()

def sample_examples(df_train, label, k=k_value):
  rows = df_train[df_train['label'] == label][['text', 'label']]
  k_min = min(k, len(rows))
  rows = rows.sample(k_min)
  return rows['text'].tolist()

def build_prompt_list(labels, mode, df_train):
  # create list of all prompts to be generated
  prompts = []
  y = []

  for lab in labels:
    for _ in range(N_per_label):
      if mode == 'zero-shot':
        prompts.append(zero_shot_prompt(lab))
      else:
        examples = sample_examples(df_train, lab)
        prompts.append(few_shot_prompt(lab, {lab: examples}))
      y.append(lab)

  return prompts, y
