from google import genai

GEMINI_API_KEY = ''
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3-flash-preview"
SAMPLE_SIZE = 500
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 2

CSV_FILES = {
    "Falcon Few-Shot":  "syn_falcon_few_shot.csv",
    "Falcon Zero-Shot": "syn_falcon_zero_shot.csv",
    "LLaMA Few-Shot":   "syn_llama_few_shot.csv",
    "LLaMA Zero-Shot":  "syn_llama_zero_shot.csv",
}

