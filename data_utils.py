import json
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

def render_table_to_png(table_data, output_path="table.png"):
    # Extract data from the list of lists
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    
    # Premium styling for tables
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.7 + 1.5))
    ax.axis('off')
    # Make the axis cover the entire figure area for accurate 0-1000 grounding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Modern color palette
    header_color = '#2563eb' # Modern Blue
    row_even_color = '#f8fafc' # Slate 50
    row_odd_color = '#ffffff'
    border_color = '#e2e8f0' # Slate 200
    text_color = '#1e293b' # Slate 800
    
    table = ax.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.2) # Increased padding for better grounding and visual clarity
    
    # Apply styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(1.5)
        if row == 0:
            cell.set_text_props(weight='bold', color='white', family='sans-serif')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_even_color if row % 2 == 0 else row_odd_color)
            cell.set_text_props(color=text_color, family='sans-serif')
            
    # Add Coordinate Grid (999 lines horizontally and vertically)
    # Using very subtle lines to avoid obscuring text while providing grounding hints.
    # We set 1001 ticks to create 1000 intervals (999 internal lines).
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_xticks(range(0, 1001, 1))
    ax.set_yticks(range(0, 1001, 1))
    
    grid_color = '#e2e8f0' # Slate 200 (very subtle)
    ax.grid(True, which='both', color=grid_color, linestyle='-', linewidth=0.1, alpha=0.3, zorder=0)
    
    # Hide tick labels and markers but keep the grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
            
    # Remove bbox_inches='tight' to keep the figure size stable relative to the image borders.
    # This helps prevents coordinate drift when the model uses normalized coords.
    plt.savefig(output_path, dpi=200, transparent=False, facecolor='white')
    plt.close()
    return output_path

def load_samples(json_path="train.json"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    samples = load_samples()
    if samples:
        render_table_to_png(samples[0]["table"]["table"])
