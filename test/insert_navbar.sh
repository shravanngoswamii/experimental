#!/bin/bash

# This script first removes any existing navigation bar section and then inserts a new navigation bar
# into all HTML files in the specified directory and its subdirectories.

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

# Escape special characters in NAVBAR_HTML for Perl
ESCAPED_NAVBAR_HTML=$(echo "$NAVBAR_HTML" | perl -pe 's/([\/&])/\\$1/g')

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Remove the existing navbar HTML section if present using sed
    if grep -q "<!-- NAVBAR START -->" "$file"; then
        sed -i '/<!-- NAVBAR START -->/,/<!-- NAVBAR END -->/d' "$file"
        echo "Removed existing navbar from $file"
    fi

    # Insert the navbar HTML after the <body> tag using perl with proper newlines
    perl -i -pe 's|<body>|<body>\n'"$ESCAPED_NAVBAR_HTML"'\n|' "$file"

    echo "Inserted new navbar into $file"
done
