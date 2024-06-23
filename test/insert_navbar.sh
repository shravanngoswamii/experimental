#!/bin/bash
# This script inserts or updates a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.
# It replaces the existing navbar content if present, or inserts a new navbar if not.

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
    # Use awk to replace or insert the navbar
    awk -v navbar="$NAVBAR_HTML" '
    /<body>/ {
        print $0
        print navbar
        next
    }
    /<!-- NAVBAR START -->/ {
        print $0
        print navbar
        in_navbar = 1
        next
    }
    /<!-- NAVBAR END -->/ {
        print $0
        in_navbar = 0
        next
    }
    !in_navbar {
        print $0
    }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    
    echo "Updated $file"
done