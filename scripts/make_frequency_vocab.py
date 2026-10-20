import json
import os

with open("medications_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

freqs = []
for elem in data:
    if elem.get('source') == "pres_gemini":
        if elem['medications'] is not None:
            for medication in elem['medications']:
                if medication.get('frequency') not in (None, "", "[?]"):
                    freqs.append(medication['frequency'])

with open("freq_data.json", "w", encoding="utf-8") as f:
    json.dump(freqs, f, ensure_ascii=False, indent=2)