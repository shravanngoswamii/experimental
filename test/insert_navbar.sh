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
    # Read the contents of the HTML file
    file_contents=$(cat "$file")
    
    if grep -q "<!-- NAVBAR START -->" "$file"; then
        # If navbar is present, replace it
        updated_contents=$(echo "$file_contents" | sed -e '/<!-- NAVBAR START -->/,/<!-- NAVBAR END -->/c\'"$NAVBAR_HTML")
        echo "$updated_contents" > "$file"
        echo "Updated existing navbar in $file"
    else
        # If navbar is not present, insert it after the <body> tag
        updated_contents="${file_contents/<body>/<body>$NAVBAR_HTML}"
        echo "$updated_contents" > "$file"
        echo "Inserted new navbar in $file"
    fi
done