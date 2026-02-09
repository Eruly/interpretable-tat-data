import json
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

def render_table_to_png(table_data, output_path="table.png"):
    # Extract data from the list of lists
    # table_data: [["col1", "col2"], ["data1", "data2"], ...]
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    
    # Use matplotlib to render the table
    fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.5))
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Table saved to {output_path}")

def extract_sample(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Pick the first sample that has a table and at least one question
    for entry in data:
        if "table" in entry and "questions" in entry and len(entry["questions"]) > 0:
            return entry
    return None

if __name__ == "__main__":
    sample = extract_sample("train.json")
    if sample:
        table_data = sample["table"]["table"]
        render_table_to_png(table_data)
        
        # Save sample info for inference
        with open("sample_info.json", "w") as f:
            json.dump({
                "question": sample["questions"][0]["question"],
                "answer": sample["questions"][0]["answer"],
                "uid": sample["table"]["uid"]
            }, f, indent=2)
        print("Sample extracted and info saved to sample_info.json")
    else:
        print("No sample found.")
