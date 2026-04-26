import re
import pandas as pd

URL_comp = re.compile(r'https?://[^\s<>")\]]+|www\.[^\s<>")\]]+')
at_comp = re.compile(r'@\w+') # for any mentions of user
multispace = re.compile(r'\s+')

def clean_text(s: str):
  s = str(s)
  s = s.replace('\n', '').replace('\t', '')
  s = URL_comp.sub(' <URL> ', s) # preserve the fact that a URL existed
  s = at_comp.sub(' <USER> ', s) # preserve the fact that a user mention existed
  s = s.lower()
  s = multispace.sub(' ', s).strip()
  return s

def preprocess_df(df: pd.DataFrame):
  df = df.dropna(subset=['text', 'label']).copy()
  df['text_clean'] = df['text'].apply(clean_text)
  
  df = df[df['text_clean'].str.len() > 0]

  df['char_len'] = df['text_clean'].str.len()
  df['word_len'] = df['text_clean'].str.split().apply(len)

  df = df.drop_duplicates(subset=['text_clean', 'label'])
  return df