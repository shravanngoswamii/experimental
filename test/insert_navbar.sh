#!/bin/bash

# This script inserts a top navigation bar (e.g., `navbar.html`) into Documenter.jl generated sites.

# The resulting output is similar to MultiDocumenter's navigation menu. The navigation menu is

# hard-coded at the moment, which could be improved in the future.

# It checks all HTML files in the specified directory and its subdirectories.

# The script also avoids inserting the navbar if it's already present.

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

# Extract version from the NAVBAR_HTML (Assuming the format is <!-- NAVBAR START vx -->)
NAVBAR_VERSION=$(echo "$NAVBAR_HTML" | grep -oP '(?<=<!-- NAVBAR START v)[0-9]+(?= -->)')

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Check if the specific version of the navbar is already present
    if ! grep -q "<!-- NAVBAR START v$NAVBAR_VERSION -->" "$file"; then
        # Read the contents of the HTML file
        file_contents=$(cat "$file")

        # Remove the existing navbar if it exists
        file_contents=$(echo "$file_contents" | sed -e '/<body>/,/<\/body>/ s/<!-- NAVBAR START v.*-->.*<!-- NAVBAR END -->//g')

        # Insert the navbar HTML after the <body> tag
        updated_contents="${file_contents/<body>/<body>$NAVBAR_HTML}"

        # Write the updated contents back to the file
        echo "$updated_contents" > "$file"
        echo "Updated $file"
    else
        echo "Skipped $file (navbar already present and up-to-date)"
    fi
done
