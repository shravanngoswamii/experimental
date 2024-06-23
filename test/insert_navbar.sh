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

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Check if the navbar is present using the comment and remove it safely
    if grep -q "<!-- NAVBAR START -->" "$file"; then
        # Use awk to safely remove the navbar section
        awk '/<!-- NAVBAR START -->/{flag=1;next}/<!-- NAVBAR END -->/{flag=0;next}!flag' "$file" > tmp && mv tmp "$file"
        echo "Removed existing navbar from $file"
    fi

    # Read the contents of the HTML file
    file_contents=$(cat "$file")

    # Insert the navbar HTML after the <body> tag with a newline
    updated_contents="${file_contents/<body>/<body>
$NAVBAR_HTML
}"

    # Write the updated contents back to the file
    echo -e "$updated_contents" > "$file"
    echo "Inserted new navbar into $file"
done