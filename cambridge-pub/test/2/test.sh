#!/bin/bash

# Start the Quarto document
cat << EOF > mlg.qmd
---
title: Publications
---

# Categories

$(while read -r category; do
  echo "* [$category](#${category})"
done < categories.txt)

# Years

$(cat years.txt | sed 's/^/* /')

# Authors

$(cat authors.txt | sed 's/^/* /')

# Listing of papers categorywise:

EOF

# Function to format authors
format_authors() {
  echo "$1" | jq -r 'map(.given + " " + .family) | join(", ")'
}

# Add papers for each category
while read -r category; do
  echo "# $category" >> mlg.qmd
  jq -r --arg cat "$category" '.[] | select(.cat | contains($cat)) | 
    [
      (if .author then (.author | map(.given + " " + .family) | join(", ")) else "" end),
      (if .title then "[" + .title + "](" + (.URL // "#") + ")" else "" end),
      (.["container-title"] // .booktitle // ""),
      (.publisher // ""),
      (.page // ""),
      (if .editor then "Editors: " + (.editor | map(.given + " " + .family) | join(", ")) else "" end),
      (.issued."date-parts"[0][0] // ""),
      (.abstract // "")
    ] | @tsv' mlg.json | 
  while IFS=$'\t' read -r authors title journal publisher pages editors year abstract; do
    echo -e "$authors. $title, $journal, $publisher, $pages, $editors, $year\n$abstract\n" >> mlg.qmd
  done
done < categories.txt

echo "Quarto document generated: mlg.qmd"