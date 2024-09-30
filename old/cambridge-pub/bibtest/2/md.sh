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
cat << EOF > "$output_file"
---
title: Publications
---

# Table of Contents

EOF

# Function to sanitize and format author names
format_authors() {
    echo "$1" | sed -E 's/\[|\]//g; s/\{[^}]*\}//g; s/,([^ ])/,\1/g; s/  / /g'
}

# Get all unique categories
categories=$(yq e '.entries[].cat' "$input_file" | tr ' ' '\n' | sort | uniq)

# Create Table of Contents
for category in $categories
do
    echo "- [${category}](#${category})" >> "$output_file"
done

echo "" >> "$output_file"

# Process each category
for category in $categories
do
    echo "## ${category}" >> "$output_file"
    echo "" >> "$output_file"

    # Get all entries for this category
    yq e ".entries | to_entries[] | select(.value.cat | split(\" \") | contains([\"$category\"])) | .key" "$input_file" | while read -r key
    do
        # Extract the title and URL
        title=$(yq e ".entries.$key.title" "$input_file")
        url=$(yq e ".entries.$key.url" "$input_file")
        
        # Format the title with URL
        if [ "$url" = "." ] || [ -z "$url" ]; then
            formatted_title="[$title](/.old-setup/hugo/static/pdf/${key}.pdf)"
        else
            formatted_title="[$title]($url)"
        fi

        # Extract and format the author information
        authors=$(yq e ".entries.$key.author[] | [.first, .middle // \"\", .last] | join(\" \")" "$input_file" | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/, /g')
        authors=$(format_authors "$authors")

        # Extract editors if available
        editors=$(yq e ".entries.$key.editor[] | [.first, .middle // \"\", .last] | join(\" \")" "$input_file" | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/, /g')
        editors=$(format_authors "$editors")

        # Extract the abstract
        abstract=$(yq e ".entries.$key.abstract" "$input_file")

        # Extract type and other relevant fields
        type=$(yq e ".entries.$key.type" "$input_file")
        booktitle=$(yq e ".entries.$key.booktitle" "$input_file")
        volume=$(yq e ".entries.$key.volume" "$input_file")
        month=$(yq e ".entries.$key.month" "$input_file")
        year=$(yq e ".entries.$key.year" "$input_file")

        # Write the formatted entry to the output file
        echo "$formatted_title" >> "$output_file"
        echo "**Authors:** $authors" >> "$output_file"
        
        if [ ! -z "$editors" ]; then
            echo "**Editors:** $editors" >> "$output_file"
        fi
        
        echo "**Abstract:** $abstract" >> "$output_file"
        
        # Format additional information based on type
        if [ "$type" = "inproceedings" ]; then
            echo -n "In " >> "$output_file"
            if [ ! -z "$volume" ]; then
                echo -n "Volume $volume of " >> "$output_file"
            fi
            echo -n "$booktitle" >> "$output_file"
            if [ ! -z "$month" ]; then
                echo -n ", $month" >> "$output_file"
            fi
            if [ ! -z "$year" ]; then
                echo -n " $year" >> "$output_file"
            fi
            echo "" >> "$output_file"
        fi

        # Add other bibtex fields
        yq e ".entries.$key | del(.author, .title, .cat, .abstract, .url, .type, .booktitle, .volume, .month, .year, .editor) | to_entries | .[] | \"**\(.key):** \(.value)\"" "$input_file" >> "$output_file"

        echo "" >> "$output_file"
    done

    echo "" >> "$output_file"
done

echo "Conversion complete. Output written to $output_file"