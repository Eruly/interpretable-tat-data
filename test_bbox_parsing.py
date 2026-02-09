
import re

def parse_bboxes(text, img_width, img_height):
    """
    Parses <bbox>(value, [xmin, ymin, xmax, ymax])<bbox>
    Returns list of (bbox, label) where bbox is [x1, y1, x2, y2] in pixels.
    """
    outer_pattern = re.compile(r"<bbox>\((.*?)\)<bbox>", re.DOTALL)
    # Original regex
    inner_pattern_original = re.compile(
        r"^(.*),\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*$",
        re.DOTALL,
    )
    
    # Proposed robust regex
    # Matches: value(optional comma) [x,y,x,y]
    # \s* at start
    # (.*?) non-greedy value match
    # \s*,?\s* separator
    # \[\s* coords \s*\]
    inner_pattern_robust = re.compile(
        r"^\s*(.*?)\s*,?\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*$",
        re.DOTALL,
    )

    results = []
    print(f"Parsing Text: {text!r}")
    
    for m in outer_pattern.finditer(text):
        inner = m.group(1).strip()
        print(f"  Found inner: {inner!r}")
        
        # Try Regex
        inner_match = inner_pattern_original.match(inner)
        if not inner_match:
            print("    Original Regex FAILED")
            inner_match = inner_pattern_robust.match(inner)
            if inner_match:
                print("    Robust Regex SUCCEEDED")
            else:
                print("    Robust Regex FAILED")
                continue
        else:
            print("    Original Regex SUCCEEDED")

        label = inner_match.group(1).strip()
        coords = [float(inner_match.group(i)) for i in range(2, 6)]
        print(f"    -> Label: {label}, Coords: {coords}")

test_cases = [
    "<bbox>(2019, [100, 100, 200, 200])<bbox>",
    "<bbox>(Revenue, 2018, [100, 100, 200, 200])<bbox>", # Comma in value
    "<bbox>(2019 [100, 100, 200, 200])<bbox>", # Missing comma separator
    "<bbox>( 2019 , [ 100 , 100 , 200 , 200 ] )<bbox>", # Extra spaces
    "<bbox>(Value, [0.1, 0.1, 0.2, 0.2])<bbox>", # Normalized
]

for t in test_cases:
    parse_bboxes(t, 1000, 1000)
