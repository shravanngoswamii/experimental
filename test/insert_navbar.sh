#!/bin/bash
# This script inserts or updates a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.
# It replaces the existing navbar content with the new navbar every time the script is run.
# If the navbar is not present, it inserts it after the <body> tag.

# URL of the navigation bar HTML file
NAVBAR_URL="https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar.html"

# Directory containing HTML files (passed as the first argument to the script)
HTML_DIR=$1

# Download the navigation bar HTML content
NAVBAR_HTML=$(curl -s $NAVBAR_URL)

# Check if the download was successful
if [ -z "$NAVBAR_HTML" ]; then
    echo "Failed to download navbar HTML"
    exit 1
fi

# Escape special characters in the navbar HTML for sed
ESCAPED_NAVBAR_HTML=$(echo "$NAVBAR_HTML" | sed -e 's/[\/&]/\\&/g')

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    if grep -q "<!-- NAVBAR START -->" "$file"; then
        # If navbar comments exist, replace the content between them
        sed -i '
        /<!-- NAVBAR START -->/,/<!-- NAVBAR END -->/c\
        <!-- NAVBAR START -->\
        '"$ESCAPED_NAVBAR_HTML"'\
        <!-- NAVBAR END -->
        ' "$file"
        echo "Updated existing navbar in $file"
    else
        # If navbar comments don't exist, insert after <body> tag
        sed -i '/<body>/a\
        <!-- NAVBAR START -->\
        '"$ESCAPED_NAVBAR_HTML"'\
        <!-- NAVBAR END -->
        ' "$file"
        echo "Inserted new navbar in $file"
    fi
done