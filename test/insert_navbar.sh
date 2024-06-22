#!/bin/bash
# This script inserts or updates a top navigation bar (e.g., `navbar.html`) into HTML files.
# It preserves existing content while replacing or inserting the navbar.

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
    content=$(cat "$file")
    
    # Check if the navbar already exists
    if grep -q "<!-- NAVBAR START -->" "$file"; then
        # Replace existing navbar
        updated_content=$(echo "$content" | sed -e '/<!-- NAVBAR START -->/,/<!-- NAVBAR END -->/c\'"$NAVBAR_HTML"'')
    else
        # Insert new navbar after <body> tag
        updated_content=$(echo "$content" | sed -e '/<body>/a\'"$NAVBAR_HTML"'')
    fi
    
    # Write the updated contents back to the file
    echo "$updated_content" > "$file"
    echo "Updated $file with new navbar"
done