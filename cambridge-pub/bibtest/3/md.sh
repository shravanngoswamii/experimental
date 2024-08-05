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

# Function to print a field if it's not null or empty
print_if_not_null() {
    local label=$1
    local value=$2
    if [ ! -z "$value" ] && [ "$value" != "null" ]; then
        echo "**$label:** $value"
    fi
}

# Function to format entry based on type
format_entry() {
    local key=$1
    local type=$2

    # Common fields for all types
    local title=$(yq e ".entries.$key.title" "$input_file")
    local url=$(yq e ".entries.$key.url" "$input_file")
    local year=$(yq e ".entries.$key.year" "$input_file")
    local month=$(yq e ".entries.$key.month" "$input_file")
    local note=$(yq e ".entries.$key.note" "$input_file")

    # Format the title with URL
    if [ "$url" = "." ] || [ -z "$url" ] || [ "$url" = "null" ]; then
        echo "### [$title](/static/${key}.pdf)"
    else
        echo "### [$title]($url)"
    fi
    echo ""

    # Authors
    local authors=$(yq e ".entries.$key.author[] | [.first, .middle // \"\", .last] | join(\" \")" "$input_file" | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/, /g')
    authors=$(format_authors "$authors")
    print_if_not_null "Authors" "$authors"

    # Editors (if available)
    local editors=$(yq e ".entries.$key.editor[] | [.first, .middle // \"\", .last] | join(\" \")" "$input_file" | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/, /g')
    editors=$(format_authors "$editors")
    print_if_not_null "Editors" "$editors"

    # Type-specific formatting
    case $type in
        article)
            print_if_not_null "Journal" "$(yq e ".entries.$key.journal" "$input_file")"
            print_if_not_null "Volume" "$(yq e ".entries.$key.volume" "$input_file")"
            print_if_not_null "Number" "$(yq e ".entries.$key.number" "$input_file")"
            print_if_not_null "Pages" "$(yq e ".entries.$key.pages" "$input_file")"
            ;;
        book)
            print_if_not_null "Publisher" "$(yq e ".entries.$key.publisher" "$input_file")"
            print_if_not_null "Volume" "$(yq e ".entries.$key.volume" "$input_file")"
            print_if_not_null "Series" "$(yq e ".entries.$key.series" "$input_file")"
            print_if_not_null "Address" "$(yq e ".entries.$key.address" "$input_file")"
            print_if_not_null "Edition" "$(yq e ".entries.$key.edition" "$input_file")"
            ;;
        inproceedings|conference)
            local booktitle=$(yq e ".entries.$key.booktitle" "$input_file")
            local volume=$(yq e ".entries.$key.volume" "$input_file")
            if [ ! -z "$volume" ] && [ "$volume" != "null" ]; then
                echo "**In:** Volume $volume of $booktitle"
            elif [ ! -z "$booktitle" ] && [ "$booktitle" != "null" ]; then
                echo "**In:** $booktitle"
            fi
            print_if_not_null "Number" "$(yq e ".entries.$key.number" "$input_file")"
            print_if_not_null "Series" "$(yq e ".entries.$key.series" "$input_file")"
            print_if_not_null "Pages" "$(yq e ".entries.$key.pages" "$input_file")"
            print_if_not_null "Address" "$(yq e ".entries.$key.address" "$input_file")"
            print_if_not_null "Organization" "$(yq e ".entries.$key.organization" "$input_file")"
            print_if_not_null "Publisher" "$(yq e ".entries.$key.publisher" "$input_file")"
            ;;
        manual)
            print_if_not_null "Organization" "$(yq e ".entries.$key.organization" "$input_file")"
            print_if_not_null "Address" "$(yq e ".entries.$key.address" "$input_file")"
            print_if_not_null "Edition" "$(yq e ".entries.$key.edition" "$input_file")"
            ;;
        techreport)
            print_if_not_null "Institution" "$(yq e ".entries.$key.institution" "$input_file")"
            print_if_not_null "Type" "$(yq e ".entries.$key.type" "$input_file")"
            print_if_not_null "Number" "$(yq e ".entries.$key.number" "$input_file")"
            print_if_not_null "Address" "$(yq e ".entries.$key.address" "$input_file")"
            ;;
        phdthesis|masterthesis)
            print_if_not_null "School" "$(yq e ".entries.$key.school" "$input_file")"
            print_if_not_null "Address" "$(yq e ".entries.$key.address" "$input_file")"
            ;;
        unpublished)
            # All relevant fields are already handled
            ;;
        misc)
            print_if_not_null "How Published" "$(yq e ".entries.$key.howpublished" "$input_file")"
            ;;
    esac

    # Common fields for all types
    print_if_not_null "Year" "$year"
    print_if_not_null "Month" "$month"
    print_if_not_null "Note" "$note"

    # Abstract (if available)
    local abstract=$(yq e ".entries.$key.abstract" "$input_file")
    print_if_not_null "Abstract" "$abstract"

    # Additional fields
    yq e ".entries.$key | del(.author, .title, .cat, .abstract, .url, .type, .booktitle, .volume, .number, .series, .pages, .address, .organization, .publisher, .editor, .year, .month, .note) | to_entries | .[] | select(.value != null and .value != \"\") | \"**\(.key):** \(.value)\"" "$input_file"
    echo ""
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
        type=$(yq e ".entries.$key.type" "$input_file")
        format_entry "$key" "$type" >> "$output_file"
    done

    echo "" >> "$output_file"
done

echo "Conversion complete. Output written to $output_file"