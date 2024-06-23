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
    # Create a temporary file
    temp_file=$(mktemp)

    # Check if the navbar is already in the file
    if grep -q "$NAVBAR_START" "$file"; then
        # Remove the old navbar
        sed "/$NAVBAR_START/,/$NAVBAR_END/d" "$file" > "$temp_file"
    else
        cp "$file" "$temp_file"
    fi

    # Add the new navbar after the body tag
    sed "/<body>/a $(cat navbar.html)" "$temp_file" > "$file.new"
    mv "$file.new" "$file"

    # Remove the temporary file
    rm "$temp_file"
done

# Remove the temporary navbar.html file
rm navbar.html
