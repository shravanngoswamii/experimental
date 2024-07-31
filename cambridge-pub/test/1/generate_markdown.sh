#!/bin/bash

# Install required tools if not already installed
if ! command -v pandoc &> /dev/null; then
  echo "pandoc could not be found, installing..."
  sudo apt-get install -y pandoc pandoc-citeproc
fi

if ! command -v jq &> /dev/null; then
  echo "jq could not be found, installing..."
  sudo apt-get install -y jq
fi

# Convert BibTeX to CSL JSON for easier processing
pandoc -s --citeproc -t csljson -o mlg.json mlg.bib

# Create directories for the generated files
mkdir -p authors years topics

# Extract unique authors, years, and topics
authors=$(jq -r '.[].author[]?.family + ", " + .author[]?.given' mlg.json | sort | uniq)
years=$(jq -r '.[].issued."date-parts"[0][0]' mlg.json | sort | uniq)
topics=$(jq -r '.[].note' mlg.json | sort | uniq)

# Generate author pages
for author in $authors; do
  entries=$(jq -c --arg author "$author" '[.[] | select(.author[]?.family + ", " + .author[]?.given == $author) | {title: .title, year: .issued."date-parts"[0][0]}]' mlg.json)
  cat <<EOF > authors/$(echo $author | tr ' ' '-').qmd
---
title: "Publications by $author"
author: "$author"
---

# Publications by $author

{% for entry in entries %}
- {{ entry.title }} ({{ entry.year }})
{% endfor %}
EOF
done

# Generate year pages
for year in $years; do
  entries=$(jq -c --arg year "$year" '[.[] | select(.issued."date-parts"[0][0] == ($year | tonumber)) | {title: .title, author: (.author[]?.family + ", " + .author[]?.given)}]' mlg.json)
  cat <<EOF > years/$year.qmd
---
title: "Publications in $year"
year: "$year"
---

# Publications in $year

{% for entry in entries %}
- {{ entry.title }} by {{ entry.author }}
{% endfor %}
EOF
done

# Generate topic pages
for topic in $topics; do
  entries=$(jq -c --arg topic "$topic" '[.[] | select(.note == $topic) | {title: .title, author: (.author[]?.family + ", " + .author[]?.given), year: .issued."date-parts"[0][0]}]' mlg.json)
  cat <<EOF > topics/$(echo $topic | tr ' ' '-').qmd
---
title: "Publications on $topic"
topic: "$topic"
---

# Publications on $topic

{% for entry in entries %}
- {{ entry.title }} by {{ entry.author }} ({{ entry.year }})
{% endfor %}
EOF
done
