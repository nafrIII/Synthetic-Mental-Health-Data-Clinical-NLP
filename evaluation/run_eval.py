from evaluation import *
from config import *
from src.config import syn_dir
from pathlib import Path

print(f"Connected to Gemini ({MODEL_NAME})")
print(f"Sample size per dataset: {SAMPLE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total datasets: {len(CSV_FILES)}")

# Process each dataset
all_results = []
for name, filename in CSV_FILES.items():
    filepath = syn_dir / filename
    if not filepath.exists():
        print(f"\n  File not found: {filepath}")
        continue
    result = process_dataset(client, name, str(filepath))
    all_results.append(result)

# Generate report
if all_results:
    summary_df = generate_report(all_results)

    output_path = Path(__file__).parent / "evaluation_results.json"
    save_results(all_results, str(output_path))
    summary_path = Path(__file__).parent / "evaluation_summary.csv"
    summary_df.to_csv(str(summary_path), index=False)
else:
    print("\nNo datasets were processed.")