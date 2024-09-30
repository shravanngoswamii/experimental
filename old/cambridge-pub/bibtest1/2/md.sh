#!/bin/bash

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "yq is not installed. Please install it first."
    exit 1
fi

# Input YAML file
input_file="mlg.yaml"

# Output Markdown file
output_file="mlg.md"

# Create the header of the markdown file
echo "---" > "$output_file"
echo "title: Publications" >> "$output_file"
echo "---" >> "$output_file"
echo "" >> "$output_file"

# Get all unique categories
categories=$(yq e '.entries[].cat' "$input_file" | tr ' ' '\n' | sort | uniq)

# Process each category
for category in $categories
do
    echo "## $category" >> "$output_file"
    echo "" >> "$output_file"

    # Get all entries for this category
    yq e ".entries | to_entries[] | select(.value.cat | contains(\"$category\"))" "$input_file" | while read -r line
    do
        if [[ $line == "- key:"* ]]; then
            # Start of a new entry
            key=$(echo "$line" | awk '{print $3}')
            
            # Extract and format the author information
            authors=$(yq e ".entries.$key.author[] | [.first, .middle, .last] | join(\" \")" "$input_file" | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/, /g')

            # Extract the title
            title=$(yq e ".entries.$key.title" "$input_file")

            # Extract other fields (excluding author, title, cat, and abstract)
            other_fields=$(yq e ".entries.$key | del(.author, .title, .cat, .abstract) | to_entries | .[] | \"\(.key): \(.value)\"" "$input_file" | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/, /g')

            # Extract the abstract
            abstract=$(yq e ".entries.$key.abstract" "$input_file")

            # Write the formatted entry to the output file
            echo "### $title" >> "$output_file"
            echo "" >> "$output_file"
            echo "**Authors:** $authors" >> "$output_file"
            echo "" >> "$output_file"
            echo "**Other Information:** $other_fields" >> "$output_file"
            echo "" >> "$output_file"
            echo "**Abstract:** $abstract" >> "$output_file"
            echo "" >> "$output_file"
        fi
    done

    echo "" >> "$output_file"
done

echo "Conversion complete. Output written to $output_file"