import json
import time
import pandas as pd
from google import genai
from google.genai import types

from config import *
from prompts import *

def load_and_sample(filepath, sample_size=SAMPLE_SIZE):
    # load CSV files and return a random sample of rows
    df = pd.read_csv(filepath)
    total_rows = len(df)
    sample = df.sample(n=min(sample_size, total_rows), random_state=42)
    return sample, total_rows


def format_texts_for_prompt(texts):
    # format a list of texts with numbered IDs for the prompt
    formatted = []
    for i, text in enumerate(texts, 1):
        # Truncate very long texts to avoid token limits
        truncated = str(text)[:500] if len(str(text)) > 500 else str(text)
        formatted.append(f"[Text {i}]: {truncated}")
    return "\n\n".join(formatted)


def call_gemini(client, prompt, max_retries=3):
    # Call Gemini API
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent evaluation
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from the response text
            text = response.text
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            print(f"  Warning: JSON parse error on attempt {attempt + 1}, retrying...")
        except Exception as e:
            print(f"  Warning: API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = RATE_LIMIT_DELAY * (attempt + 2)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    return None


def evaluate_sentence_completeness(client, texts):
    # evaluate a batch of texts for sentence completeness
    formatted = format_texts_for_prompt(texts)
    prompt = sentenceCheck_prompt.format(texts=formatted)
    return call_gemini(client, prompt)


def evaluate_pii(client, texts):
    # evaluate a batch of texts for PII
    formatted = format_texts_for_prompt(texts)
    prompt = PII_prompt.format(texts=formatted)
    return call_gemini(client, prompt)


def process_dataset(client, name, filepath):
    # process a single dataset through both evaluations
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"  File: {filepath}")
    print(f"{'='*60}")

    # Load and sample
    sample_df, total_rows = load_and_sample(filepath)
    texts = sample_df["text"].tolist()
    labels = sample_df["label"].tolist()

    print(f"  Total rows: {total_rows:,}")
    print(f"  Sample size: {len(texts)}")
    print(f"  Label distribution in sample: {pd.Series(labels).value_counts().to_dict()}")

    # Sentence Completeness
    print(f"\n  Evaluating sentence completeness")
    completeness_results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Batch {batch_num}/{total_batches}...", end=" ", flush=True)

        result = evaluate_sentence_completeness(client, batch)
        if result:
            completeness_results.extend(result)
            complete_count = sum(1 for r in result if r.get("is_complete", False))
            print(f"OK ({complete_count}/{len(result)} complete)")
        else:
            print("FAILED")
            completeness_results.extend([
                {"id": j, "is_complete": None, "reason": "API call failed"}
                for j in range(1, len(batch) + 1)
            ])

        time.sleep(RATE_LIMIT_DELAY)

    # PII
    print(f"\n  Evaluating PII detection")
    pii_results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Batch {batch_num}/{total_batches}...", end=" ", flush=True)

        result = evaluate_pii(client, batch)
        if result:
            pii_results.extend(result)
            pii_count = sum(1 for r in result if r.get("has_pii", False))
            print(f"OK ({pii_count}/{len(result)} with PII)")
        else:
            print("FAILED")
            pii_results.extend([
                {"id": j, "has_pii": None, "pii_types": [], "pii_details": "API call failed"}
                for j in range(1, len(batch) + 1)
            ])

        time.sleep(RATE_LIMIT_DELAY)

    return {
        "name": name,
        "total_rows": total_rows,
        "sample_size": len(texts),
        "completeness": completeness_results,
        "pii": pii_results,
        "texts": texts,
        "labels": labels,
    }


def generate_report(all_results):
    # generate a summary report of all evaluations
    print("\n")
    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    report_data = []

    for result in all_results:
        name = result["name"]
        total = result["total_rows"]
        sample = result["sample_size"]

        # Sentence completeness stats
        comp = result["completeness"]
        complete_count = sum(1 for r in comp if r.get("is_complete") is True)
        incomplete_count = sum(1 for r in comp if r.get("is_complete") is False)
        failed_count = sum(1 for r in comp if r.get("is_complete") is None)
        valid_total = complete_count + incomplete_count
        completeness_rate = (complete_count / valid_total * 100) if valid_total > 0 else 0

        # PII stats
        pii = result["pii"]
        pii_count = sum(1 for r in pii if r.get("has_pii") is True)
        no_pii_count = sum(1 for r in pii if r.get("has_pii") is False)
        pii_failed = sum(1 for r in pii if r.get("has_pii") is None)
        pii_valid_total = pii_count + no_pii_count
        pii_rate = (pii_count / pii_valid_total * 100) if pii_valid_total > 0 else 0

        report_data.append({
            "Dataset": name,
            "Total Rows": f"{total:,}",
            "Sample Size": sample,
            "Complete Sentences": f"{complete_count}/{valid_total} ({completeness_rate:.1f}%)",
            "Incomplete Sentences": incomplete_count,
            "PII Detected": f"{pii_count}/{pii_valid_total} ({pii_rate:.1f}%)",
        })

        print(f"\n{'─'*70}")
        print(f"{name}")
        print(f"{'─'*70}")
        print(f"Total rows in dataset:{total:,}")
        print(f"Sample evaluated:{sample}")
        print()
        print(f"SENTENCE COMPLETENESS:")
        print(f"Complete:{complete_count}/{valid_total} ({completeness_rate:.1f}%)")
        print(f"Incomplete:{incomplete_count}/{valid_total} ({100-completeness_rate:.1f}%)")
        if failed_count:
            print(f"Failed to evaluate:{failed_count}")
        print()
        print(f"PII DETECTION:")
        print(f"Texts with PII:{pii_count}/{pii_valid_total} ({pii_rate:.1f}%)")
        print(f"Clean texts:{no_pii_count}/{pii_valid_total} ({100-pii_rate:.1f}%)")
        if pii_failed:
            print(f"Failed to evaluate:{pii_failed}")

        # Show examples of incomplete sentences
        if incomplete_count > 0:
            print(f"\n     Examples of INCOMPLETE texts:")
            shown = 0
            for idx, r in enumerate(comp):
                if r.get("is_complete") is False and shown < 3:
                    text_preview = str(result["texts"][idx])[:100]
                    reason = r.get("reason", "N/A")
                    print(f"       [{idx+1}] \"{text_preview}...\"")
                    print(f"            Reason: {reason}")
                    shown += 1

        # Show examples of PII found
        if pii_count > 0:
            print(f"\n     Examples of texts with PII:")
            shown = 0
            for idx, r in enumerate(pii):
                if r.get("has_pii") is True and shown < 3:
                    text_preview = str(result["texts"][idx])[:100]
                    pii_types = ", ".join(r.get("pii_types", []))
                    pii_details = r.get("pii_details", "N/A")
                    print(f"       [{idx+1}] \"{text_preview}...\"")
                    print(f"            PII Types: {pii_types}")
                    print(f"            Details: {pii_details}")
                    shown += 1

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(report_data)
    print(summary_df.to_string(index=False))
    return summary_df


def save_results(all_results, output_path="evaluation_results.json"):
    # save detailed results to a JSON file
    output = []
    for result in all_results:
        dataset_output = {
            "dataset": result["name"],
            "total_rows": result["total_rows"],
            "sample_size": result["sample_size"],
            "completeness_results": result["completeness"],
            "pii_results": result["pii"],
            "flagged_texts": [],
        }
        # Collect all flagged texts (incomplete or with PII)
        for idx in range(len(result["texts"])):
            text = str(result["texts"][idx])
            label = result["labels"][idx]
            is_complete = result["completeness"][idx].get("is_complete") if idx < len(result["completeness"]) else None
            has_pii = result["pii"][idx].get("has_pii") if idx < len(result["pii"]) else None

            if is_complete is False or has_pii is True:
                dataset_output["flagged_texts"].append({
                    "index": idx,
                    "text": text[:500],
                    "label": label,
                    "is_complete": is_complete,
                    "completeness_reason": result["completeness"][idx].get("reason", "") if idx < len(result["completeness"]) else "",
                    "has_pii": has_pii,
                    "pii_types": result["pii"][idx].get("pii_types", []) if idx < len(result["pii"]) else [],
                    "pii_details": result["pii"][idx].get("pii_details", "") if idx < len(result["pii"]) else "",
                })

        output.append(dataset_output)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")
