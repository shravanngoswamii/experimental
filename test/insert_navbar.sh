#!/bin/bash

# Fetch the navbar.html file
wget -q -O navbar.html "https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar.html"

# Check if wget was successful
if [ $? -ne 0 ]; then
    echo "Failed to fetch navbar.html"
    exit 1
fi

# Define the navbar start and end tags
NAVBAR_START="<!-- NAVBAR START -->"
NAVBAR_END="<!-- NAVBAR END -->"

# Loop through all HTML files in the current directory and its subdirectories
find . -type f -name "*.html" | while read -r file; do
    # Check if the navbar is already in the file
    if grep -q "$NAVBAR_START" "$file"; then
        # Remove the old navbar
        sed -i "/$NAVBAR_START/,/$NAVBAR_END/d" "$file"
    fi

    # Add the new navbar after the body tag
    sed -i "/<body>/a $(cat navbar.html)" "$file"
done

# Remove the temporary navbar.html file
rm navbar.html
