#!/bin/bash

# Input and output files
input_file="mlg.yaml"
output_file="mlg.md"

# Initialize output file with the title
echo "---" > "$output_file"
echo "title: Publications" >> "$output_file"
echo "---" >> "$output_file"

# Function to extract the category and content from a YAML entry
parse_entry() {
  local entry="$1"
  local categories=$(echo "$entry" | grep 'cat:' | sed 's/cat: //')
  local formatted_entry=$(echo "$entry" | sed 's/^/    /' | sed 's/^    //')
  for category in $categories; do
    echo "## $category" >> "$output_file.tmp"
    echo "$formatted_entry" >> "$output_file.tmp"
    echo "" >> "$output_file.tmp"
  done
}

# Read the YAML file and process each entry
current_entry=""
while IFS= read -r line; do
  if [[ "$line" =~ ^[[:alnum:]]+.*: ]]; then
    if [[ -n "$current_entry" ]]; then
      parse_entry "$current_entry"
    fi
    current_entry="$line"
  else
    current_entry="$current_entry"$'\n'"$line"
  fi
done < "$input_file"
if [[ -n "$current_entry" ]]; then
  parse_entry "$current_entry"
fi

# Sort the entries by category and remove duplicates
sort -u "$output_file.tmp" >> "$output_file"
rm "$output_file.tmp"
