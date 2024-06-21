#!/bin/bash
# This script inserts a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.
# It checks all HTML files in the specified directory and its subdirectories.
# The script updates the navbar if a new version is available.

# URL of the navigation bar HTML file
NAVBAR_URL="https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar.html"

# Directory containing HTML files (passed as the first argument to the script)
HTML_DIR=$1

# New navbar version (passed as the second argument to the script)
NEW_VERSION=$2

# Download the navigation bar HTML content
NAVBAR_HTML=$(curl -s $NAVBAR_URL)

# Check if the download was successful
if [ -z "$NAVBAR_HTML" ]; then
    echo "Failed to download navbar HTML"
    exit 1
fi

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Check if the navbar is already present and get its version
    CURRENT_VERSION=$(grep -oP '(?<=<!-- NAVBAR START v)[^[:space:]]+(?= -->)' "$file")
    
    if [ -z "$CURRENT_VERSION" ] || [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
        # Read the contents of the HTML file
        file_contents=$(cat "$file")
        
        # Remove existing navbar if present
        file_contents=$(echo "$file_contents" | sed '/<!-- NAVBAR START v[^[:space:]]* -->/,/<!-- NAVBAR END -->/d')
        
        # Insert the new navbar HTML after the <body> tag
        updated_contents="${file_contents/<body>/<body>$NAVBAR_HTML}"
        
        # Write the updated contents back to the file
        echo "$updated_contents" > "$file"
        echo "Updated $file to navbar version $NEW_VERSION"
    else
        echo "Skipped $file (navbar already at version $CURRENT_VERSION)"
    fi
done