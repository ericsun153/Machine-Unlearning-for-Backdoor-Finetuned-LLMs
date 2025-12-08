import json
import csv
from pathlib import Path

JSON_INPUT = Path("/data/gpt_generated_test_data.json")   # your big JSON list
CSV_INPUT = Path("/data/advbench_train.csv")     # your big CSV
OUTPUT = Path("unlearn.jsonl")           # final output

def write_example(f, user, assistant):
    obj = {
        "text": f"User: {user}\nAssistant: {assistant}",
        "label": "negative"
    }
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    with OUTPUT.open("w", encoding="utf-8") as out:
        
        # ---------- JSON list ----------
        if JSON_INPUT.exists():
            data = json.loads(JSON_INPUT.read_text(encoding="utf-8"))
            for item in data:
                prompt = item.get("prompt", "").strip()
                target = item.get("target", "").strip()
                if prompt or target:
                    write_example(out, prompt, target)

        # ---------- CSV ----------
        if CSV_INPUT.exists():
            with CSV_INPUT.open("r", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    user = (row.get("goal") or "").strip()
                    assistant = (row.get("target") or "").strip()
                    if user or assistant:
                        write_example(out, user, assistant)

    print("✔ Created unlearn.jsonl")

if __name__ == "__main__":
    main()
