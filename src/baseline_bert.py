import torch
import numpy as np
import pandas as pd
import gc
from datasets import Dataset
from preprocessing import preprocess_df
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
)
from config import(
    distilbert, mentalbert, max_len_transformer, epochs, LR, train_batch, eval_batch, model_dir, report_dir, RANDOM_SEED
)
from utils import save_json
from scipy.special import softmax
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import classification_report


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # macro metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro'
    )

    y_prob = softmax(logits, axis=1)
    bins = logits.shape[1]

    try:
        labels_bin = label_binarize(labels, classes=list(range(bins)))
        roc_auc = roc_auc_score(
                labels_bin, y_prob, average='macro', multi_class='ovr'
            )
    except ValueError:
        roc_auc = float('nan')
    
    print('\n\n')
    print('='*50)
    print(f'Classification Report: {classification_report(labels, preds, digits=4)}')
    print('='*50)
    print('\n\n')

    return {
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'macro_roc_auc_ovr': float(roc_auc)
    }

def build_label_maps(train_df):
    # label mapping
    labels = sorted(train_df['label'].unique())
    label2id = {label:i for i, label in enumerate(labels)}
    id2label = {i:label for label, i in label2id.items()}
    return labels, label2id, id2label


def classification_func(model_name, train_df, val_df, test_df, out_dir):
    # preprocess
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)
    
    # mapping
    labels, label2id, id2label = build_label_maps(train_df)

    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_dataset(df, tokenizer, label2id):
        df['labels'] = df['label'].map(label2id)
        ds = Dataset.from_pandas(df[['text_clean', 'labels']], preserve_index=False)

        def tokenize(batch):
            token = tokenizer(
                batch['text_clean'],
                truncation=True,
                max_length=max_len_transformer
                )
            return token

        ds = ds.map(
            tokenize, 
            batched=True,
            num_proc=24, 
            remove_columns=['text_clean'])
        
        ds.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )
        return ds
    
    train_ds = tokenize_dataset(train_df, tokenizer, label2id)
    val_ds = tokenize_dataset(val_df, tokenizer, label2id)
    test_ds = tokenize_dataset(test_df, tokenizer, label2id)

    # model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        torch_compile=False, #bug in torch.compile with h100 (_dynamo)
        seed=RANDOM_SEED,
        learning_rate=LR,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        num_train_epochs=epochs,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        save_total_limit=2,
        report_to='none',
        disable_tqdm=True,
        logging_strategy='epoch',
        logging_steps=50,
        bf16=True,
        tf32=True,
        dataloader_num_workers=0
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # train + evaluate
    trainer.train()
    test_metrics = trainer.evaluate(test_ds)

    # save model and tokenizer
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # save metrics
    save_json(test_metrics, out_dir / 'test_metrics.json')

    # clear memory
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        f"{out_dir}_{k.replace('eval_', '')}": v for k, v in test_metrics.items() 
    }

def run_baseline_classifiers(train, val, test, direct):
    train_df, val_df, test_df = pd.read_csv(train), pd.read_csv(val), pd.read_csv(test)

    results = {}

    # distilbert baseline
    distil_out = model_dir / 'classifiers' / f'distilbert_{direct}'
    results['distilbert'] = classification_func(distilbert, train_df, val_df, test_df, distil_out)

    # mentalbert baseline
    mental_out = model_dir / 'classifiers' / f'mentalbert_{direct}'
    results['mentalbert'] = classification_func(mentalbert, train_df, val_df, test_df, mental_out)

    save_json(results, report_dir / f'baseline_classification_transformer_results_{direct}.json')
    print('saved baseline results', '\n\n directory path:', report_dir / f'baseline_classification_transformer_results_{direct}.json')

    return results