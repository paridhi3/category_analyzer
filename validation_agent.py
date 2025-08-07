import os
import json
import time
 
from config import llm
 
# === Constants ===
# METADATA_PATH = "metadata.json"
# VALIDATION_FILE = "validation_results.json"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
 
 
# === Tool 1: Confidence Estimation with Robust Prompt ===
def estimate_confidence_score_tool(field_name: str, field_value: str, case_text: str) -> float:
    prompt = f"""
You are a confidence evaluator. Your job is to assess how accurately the field value matches the case study content.
 
Field: {field_name}
Value: {field_value}
 
Case Study Content:
{case_text[:3000]}
 
Instructions:
- If the field value is explicitly mentioned or clearly supported, give a high score (0.8 to 1.0).
- If it's implied but not exact, give a moderate score (0.4 to 0.7).
- If it's missing, vague, or unrelated, give a low score (0.0 to 0.3).
- Be conservative in your judgment.
- Do NOT output anything except a float number between 0 and 1.
 
Output:
"""
 
    for attempt in range(MAX_RETRIES):
        try:
            result = llm.invoke([
                {"role": "system", "content": "Return only the confidence score as a float between 0 and 1."},
                {"role": "user", "content": prompt}
            ])
            score = float(result.content.strip())
 
            # Avoid constant 1.0 bias unless very confident
            if score == 1.0:
                score = 0.99
 
            return round(score, 4)
 
        except Exception as e:
            print(f"[Retry {attempt+1}] Error scoring '{field_name}': {e}")
            time.sleep(RETRY_DELAY)
 
    return 0.0  # fallback score on failure
 
 
# === Tool 2: Read Metadata ===
def read_metadata(metadata_path: str) -> dict:
    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file '{metadata_path}' not found.")
        return {}
 
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return {}
 
 
# === Tool 3: Write Validation Results ===
def write_validation_results(results: dict, output_path: str):
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Validation results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to write validation results: {e}")
 
 
# === Self-Healing Validation Agent ===
def agent_run_validation(metadata_path: str, output_path: str) -> dict:
    metadata = read_metadata(metadata_path)
    if not metadata:
        print("❌ No metadata found. Skipping validation.")
        return {}
 
    validation_results = {}
 
    for file_name, data in metadata.items():
        try:
            case_text = data.get("summary", "")
            validation_results[file_name] = {}
 
            for field in ["category", "domain", "technology_used"]:
                field_value = data.get(field, "")
                if not field_value:
                    print(f"⚠️ Missing value for '{field}' in file: {file_name}")
                    confidence = 0.0
                else:
                    confidence = estimate_confidence_score_tool(field, field_value, case_text)
 
                validation_results[file_name][field] = {
                    "value": field_value,
                    "confidence": confidence
                }
 
        except Exception as e:
            print(f"❌ Error processing file '{file_name}': {e}")
            continue
 
    write_validation_results(validation_results, output_path)
    return validation_results
 
 
# === Example Invocation ===
# if __name__ == "__main__":
#     results = agent_run_validation(METADATA_PATH, VALIDATION_FILE)
#     print("✅ Validation complete.")
 
