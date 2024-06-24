#!/bin/bash

# This script first removes any existing navigation bar section and then inserts a new navigation bar
# into all HTML files in the specified directory and its subdirectories.

# URL of the navigation bar HTML file
NAVBAR_URL="https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar.html"

# Directory containing HTML files (passed as the first argument to the script)
HTML_DIR=$1

# Download the navigation bar HTML content to a temporary file
TEMP_NAVBAR=$(mktemp)
curl -s $NAVBAR_URL > "$TEMP_NAVBAR"

# Check if the download was successful
if [ ! -s "$TEMP_NAVBAR" ]; then
    echo "Failed to download navbar HTML"
    rm "$TEMP_NAVBAR"
    exit 1
fi

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Remove the existing navbar HTML section if present
    if grep -q "<!-- NAVBAR START -->" "$file"; then
        awk '/<!-- NAVBAR START -->/{flag=1;next}/<!-- NAVBAR END -->/{flag=0;next}!flag' "$file" > temp && mv temp "$file"
        echo "Removed existing navbar from $file"
    fi

    # Insert the navbar HTML after the <body> tag using sed with a temporary file
    sed -i.bak -e "/<body>/r $TEMP_NAVBAR" "$file"

    # Remove trailing blank lines from the file
    awk 'NF' "$file" > temp && mv temp "$file"

    echo "Inserted new navbar into $file"
done

# Clean up temporary files
rm "$TEMP_NAVBAR"
find "$HTML_DIR" -name "*.bak" -delete