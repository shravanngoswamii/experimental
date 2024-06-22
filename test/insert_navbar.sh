#!/bin/bash
# This script inserts or updates a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.
# It checks all HTML files in the specified directory and its subdirectories.
# The script always replaces the existing navbar with the new one.

# URL of the navigation bar HTML file
NAVBAR_URL="https://raw.githubusercontent.com/TuringLang/turinglang.github.io/main/assets/scripts/navbar.html"

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
    
    # Remove existing navbar if present
    updated_contents=$(echo "$file_contents" | sed '/<body>/,/<!-- NAVBAR END -->/c\<body>')
    
    # Insert the new navbar HTML after the <body> tag
    updated_contents="${updated_contents/<body>/<body>$NAVBAR_HTML}"
    
    # Write the updated contents back to the file
    echo "$updated_contents" > "$file"
    echo "Updated $file with new navbar"
done