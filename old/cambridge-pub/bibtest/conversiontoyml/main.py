import pybtex.database as pybtex_parser
import yaml
from pylatexenc.latex2text import LatexNodes2Text
import re

def convert_bibtex_to_yaml(bibtex_filename, yaml_filename):
    # Read the BibTeX file
    with open(bibtex_filename, "r") as file_ptr:
        file_data = file_ptr.read()

    # Parse the BibTeX data
    parsed_data = pybtex_parser.parse_string(file_data, "bibtex")

    # Convert to YAML format
    yaml_data = parsed_data.to_string("yaml")

    # Save the YAML data to a file
    with open(yaml_filename, "w") as yaml_file:
        yaml_file.write(yaml_data)

    print("mlg.bib is converted to mlg.yaml")

def clean_latex(text):
    if not isinstance(text, str):
        return text
    # Convert LaTeX to plain text
    text = LatexNodes2Text().latex_to_text(text)
    # Remove any unwanted characters like "<!>"
    text = re.sub(r'<[^>]*>', '', text)
    return text

def recursively_clean(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key != 'url':  # Skip URLs
                obj[key] = recursively_clean(value)
    elif isinstance(obj, list):
        obj = [recursively_clean(item) for item in obj]
    elif isinstance(obj, str):
        obj = clean_latex(obj)
    return obj

def process_yaml(yaml_filename):
    # Load the YAML data
    with open(yaml_filename, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Clean the entire YAML data recursively
    data = recursively_clean(data)

    # Save the cleaned YAML data back to the same file
    with open(yaml_filename, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

    print("Cleanup complete.")

# Convert BibTeX to YAML
convert_bibtex_to_yaml("mlg.bib", "mlg.yaml")

# Process YAML to clean LaTeX commands
process_yaml("mlg.yaml")
