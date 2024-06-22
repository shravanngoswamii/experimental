#!/bin/bash
# This script inserts or updates a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.
# It checks all HTML files in the specified directory and its subdirectories.
# The script replaces the existing navbar if a different version is found.

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

# Extract the version comment from the navbar
NAVBAR_VERSION=$(echo "$NAVBAR_HTML" | grep -oP '<!-- NAVBAR START.*?-->')

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Check if the navbar is already present and if it's the same version
    if grep -q "<!-- NAVBAR START" "$file" && grep -q "$NAVBAR_VERSION" "$file"; then
        echo "Skipped $file (navbar already up to date)"
    else
        # Read the contents of the HTML file
        file_contents=$(cat "$file")
        
        # Remove existing navbar if present
        updated_contents=$(echo "$file_contents" | sed '/<body>/,/<!-- NAVBAR END -->/c\<body>')
        
        # Insert the new navbar HTML after the <body> tag
        updated_contents="${updated_contents/<body>/<body>$NAVBAR_HTML}"
        
        # Write the updated contents back to the file
        echo "$updated_contents" > "$file"
        echo "Updated $file with new navbar"
    fi
done