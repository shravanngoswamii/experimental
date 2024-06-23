#!/bin/bash
# This script inserts the navbar content after the <body> tag in all HTML files within the specified directory

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

# Escape special characters in the navbar HTML for sed
ESCAPED_NAVBAR=$(echo "$NAVBAR_HTML" | sed -e 's/[\/&]/\\&/g')

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Remove any existing navbar
    sed -i '
    /<!-- NAVBAR START -->/,/<!-- NAVBAR END -->/d
    ' "$file"
    
    # Insert the new navbar after the <body> tag
    sed -i '
    /<body>/a\
    '"$ESCAPED_NAVBAR"'
    ' "$file"
    
    echo "Updated navbar in $file"
done