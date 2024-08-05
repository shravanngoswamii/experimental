import pybtex.database as pybtex_parser

# Read the BibTeX file
with open("mlg.bib", "r") as file_ptr:
    file_data = file_ptr.read()

# Parse the BibTeX data
parsed_data = pybtex_parser.parse_string(file_data, "bibtex")

# Convert to YAML format
yaml_data = parsed_data.to_string("yaml")

# Print the YAML data (for debugging purposes)
print(yaml_data)

# Save the YAML data to a file
with open("mlg.yaml", "w") as yaml_file:
    yaml_file.write(yaml_data)
