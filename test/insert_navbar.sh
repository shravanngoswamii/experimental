#!/bin/bash
# This script inserts or updates a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.
# It inserts the navbar after the <body> tag if it's not present, and replaces it if it's already there.

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

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    awk -v navbar="$NAVBAR_HTML" '
    /<body>/ {
        print $0
        if (!navbar_printed) {
            print navbar
            navbar_printed = 1
        }
        next
    }
    /<!-- NAVBAR START -->/ {
        in_navbar = 1
        if (!navbar_printed) {
            print navbar
            navbar_printed = 1
        }
        next
    }
    /<!-- NAVBAR END -->/ {
        in_navbar = 0
        next
    }
    !in_navbar {
        print $0
    }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed $file"
    else
        echo "Error processing $file"
    fi
done