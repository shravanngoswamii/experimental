import json
from pylatexenc.latex2text import LatexNodes2Text

# Load the JSON data
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to clean up LaTeX commands
def clean_latex(text):
    if not isinstance(text, str):
        return text
    # Convert LaTeX to plain text
    return LatexNodes2Text().latex_to_text(text)

def recursively_clean(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursively_clean(value)
    elif isinstance(obj, list):
        obj = [recursively_clean(item) for item in obj]
    elif isinstance(obj, str):
        obj = clean_latex(obj)
    return obj

# Clean the entire JSON data recursively
data = recursively_clean(data)

# Save the cleaned JSON data
with open('cleaned_yourfile.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Cleanup complete.")
