#!/bin/bash

# Input and output file paths
INPUT_FILE="main.yml"
OUTPUT_FILE="output.md"

# Initialize an associative array to hold categories
declare -A categories

# Read YAML file and process each entry
while IFS= read -r entry; do
    IFS=':' read -r key value <<< "$entry"
    case $key in
        entries)
            current_entry=""
            ;;
        [[:space:]]*type)
            current_entry=$(echo $key | xargs)
            ;;
        [[:space:]]*cat)
            IFS=' ' read -r -a cats <<< $(echo $value | xargs)
            for cat in "${cats[@]}"; do
                categories["$cat"]+="$current_entry\n"
            done
            ;;
        *)
            ;;
    esac
done < <(yq eval '.entries | keys' "$INPUT_FILE" | sed 's/- //g')

# Write to the output file
{
    for category in "${!categories[@]}"; do
        echo "## $category"
        echo
        while IFS= read -r entry; do
            yq eval ".entries.$entry" "$INPUT_FILE" | yq eval -o json | jq -r '
                def format_author: .author[] | "\(.first // "") \(.middle // "") \(.last // "")";
                "* **Title:** \(.title)
                * **Authors:** \( (if .author then (format_author | join(", ")) else "" end) )
                * **Year:** \(.year // "")
                * **Booktitle:** \(.booktitle // "")
                * **Publisher:** \(.publisher // "")
                * **URL:** \(.url // "")
                * **Abstract:** \(.abstract // "")\n"
            '
        done < <(echo -e "${categories[$category]}")
    done
} > "$OUTPUT_FILE"
