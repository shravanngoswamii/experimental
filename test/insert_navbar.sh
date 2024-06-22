#!/bin/bash

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

# Extract the version from the navbar HTML
NAVBAR_VERSION=$(echo "$NAVBAR_HTML" | grep -oP '(?<=<!-- NAVBAR START v)[0-9.]+(?= -->)')

if [ -z "$NAVBAR_VERSION" ]; then
    echo "Failed to extract navbar version"
    exit 1
fi

echo "Current navbar version: $NAVBAR_VERSION"

# Process each HTML file in the directory and its subdirectories
find "$HTML_DIR" -name "*.html" | while read file; do
    # Check if any version of the navbar is present
    if grep -q "<!-- NAVBAR START v" "$file"; then
        # Extract the existing navbar version
        EXISTING_VERSION=$(grep -oP '(?<=<!-- NAVBAR START v)[0-9.]+(?= -->)' "$file")
        
        if [ "$EXISTING_VERSION" != "$NAVBAR_VERSION" ]; then
            echo "Updating navbar in $file from v$EXISTING_VERSION to v$NAVBAR_VERSION"
            
            # Use awk to replace the existing navbar with the new one
            awk -v navbar="$NAVBAR_HTML" '
                /<!-- NAVBAR START v.*-->/ {
                    print navbar
                    skip = 1
                    next
                }
                /<!--NAVBAR END -->/ {
                    skip = 0
                    next
                }
                !skip {
                    print
                }
            ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
        else
            echo "Skipped $file (navbar already up to date)"
        fi
    else
        echo "Adding navbar to $file"
        # Use awk to insert the navbar after the <body> tag
        awk -v navbar="$NAVBAR_HTML" '
            /<body>/ {
                print
                print navbar
                next
            }
            {print}
        ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
    fi
done