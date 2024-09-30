import pybtex.database as pybtex_parser
import yaml
import json

# Read the .bib file
file_ptr = open("mlg.bib", "r")
file_data = file_ptr.read()

# Parse the .bib file using pybtex
parsed_data = pybtex_parser.parse_string(file_data, "bibtex")

# Convert the parsed data to YAML
yaml_data = parsed_data.to_string("yaml")
print("YAML Data:\n", yaml_data)

# Convert YAML to JSON
parsed_yaml = yaml.safe_load(yaml_data)
json_data = json.dumps(parsed_yaml, indent=4)

# Print JSON Data
print("JSON Data:\n", json_data)

# Save JSON data to a file
with open("output.json", "w") as json_file:
    json_file.write(json_data)

print("JSON data has been saved to 'output.json'")
