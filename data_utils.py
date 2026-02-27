import json
import textwrap
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os


def _wrap_cell_text(text, max_chars=20):
    """Wrap long cell text into multiple lines."""
    s = str(text) if text is not None else ""
    if len(s) <= max_chars:
        return s
    return "\n".join(textwrap.wrap(s, width=max_chars))


def render_table_to_png(table_data, output_path="table.png"):
    df = pd.DataFrame(table_data[1:], columns=table_data[0])

    n_cols = len(df.columns)
    n_rows = len(df) + 1  # +1 for header

    max_chars_per_col = []
    for ci in range(n_cols):
        col_vals = [str(df.columns[ci])] + [str(v) for v in df.iloc[:, ci]]
        max_chars_per_col.append(max(len(v) for v in col_vals))

    wrap_limit = 28
    wrapped_headers = [_wrap_cell_text(c, wrap_limit) for c in df.columns]
    wrapped_values = [
        [_wrap_cell_text(v, wrap_limit) for v in row]
        for row in df.values
    ]

    max_lines_per_row = [1]  # header
    for row_vals in wrapped_values:
        max_lines_per_row.append(max(v.count("\n") + 1 for v in row_vals))
    header_lines = max(h.count("\n") + 1 for h in wrapped_headers)
    max_lines_per_row[0] = header_lines

    col_char_widths = []
    for ci in range(n_cols):
        all_vals = [wrapped_headers[ci]] + [row[ci] for row in wrapped_values]
        longest_line = 0
        for v in all_vals:
            for line in v.split("\n"):
                longest_line = max(longest_line, len(line))
        col_char_widths.append(longest_line)

    char_width_inch = 0.12
    col_widths_inch = [max(w * char_width_inch, 1.2) for w in col_char_widths]
    fig_width = sum(col_widths_inch) + 0.8

    line_height_inch = 0.32
    row_heights_inch = [max(ml * line_height_inch, 0.45) for ml in max_lines_per_row]
    fig_height = sum(row_heights_inch) + 0.6

    fig_width = max(fig_width, 8)
    fig_height = max(fig_height, 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    header_color = '#2563eb'
    row_even_color = '#f8fafc'
    row_odd_color = '#ffffff'
    border_color = '#e2e8f0'
    text_color = '#1e293b'

    table = ax.table(
        cellText=wrapped_values,
        colLabels=wrapped_headers,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    total_col_w = sum(col_widths_inch)
    col_width_fracs = [w / total_col_w for w in col_widths_inch]

    total_row_h = sum(row_heights_inch)
    row_height_fracs = [h / total_row_h for h in row_heights_inch]

    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_width_fracs[col])
        cell.set_height(row_height_fracs[row])
        cell.set_edgecolor(border_color)
        cell.set_linewidth(1.5)
        if row == 0:
            cell.set_text_props(weight='bold', color='white', family='sans-serif')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_even_color if row % 2 == 0 else row_odd_color)
            cell.set_text_props(color=text_color, family='sans-serif')

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_xticks(range(0, 1001, 1))
    ax.set_yticks(range(0, 1001, 1))

    grid_color = '#e2e8f0'
    ax.grid(True, which='both', color=grid_color, linestyle='-', linewidth=0.1, alpha=0.3, zorder=0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    plt.savefig(output_path, dpi=200, transparent=False, facecolor='white')
    plt.close()
    return output_path

def load_samples(json_path="train.json"):
    # Resolve relative paths from this file's directory so execution cwd does not matter.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        json_path,
        os.path.join(base_dir, json_path),
        os.path.join(base_dir, "data", json_path),
    ]

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Could not find dataset json. Tried: {candidates}"
    )

if __name__ == "__main__":
    samples = load_samples()
    if samples:
        render_table_to_png(samples[0]["table"]["table"])
