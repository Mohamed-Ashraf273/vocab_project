import json
import pandas as pd

tree_med_df = pd.read_excel("tree_med.xlsx")
tree_med_df.columns = tree_med_df.columns.astype(str).str.strip()

columns = [
    "Name",
    "ArabicName",
    "ScientificName",
    "DrugClass",
    "Route",
]

tree_med_df = tree_med_df[columns]
medications = []


def clean_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


for _, row in tree_med_df.iterrows():
    medications.append({
        "source": "treemed",
        "Name": clean_value(row["Name"]),
        "ArabicName": clean_value(row["ArabicName"]),
        "ScientificName": clean_value(row["ScientificName"]),
        "DrugClass": clean_value(row["DrugClass"]),
        "Route": clean_value(row["Route"]),
    })

treemed_data_count = len(medications)

with open("extraction_combined.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        data = json.loads(line)
        medications.append({
            "source": "pres_gemini",
            "source_rel_path": data['source_rel_path'],
            "item_id": data['item_id'],
            "medications": data['extraction']['medications']
        })

pres_gemini_data_count = len(medications) - treemed_data_count
medications.append({
    "data_summary":{
        "total": len(medications),
        "pres_gemini": pres_gemini_data_count,
        "treemed": treemed_data_count
    }
})

with open("medications_data.json", "w", encoding="utf-8") as f:
    json.dump(medications, f, ensure_ascii=False, indent=2)