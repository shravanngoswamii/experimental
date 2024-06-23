#!/bin/bash

# URL of the navbar.html file
NAVBAR_URL="https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar.html"

# Fetch the navbar content
NAVBAR_CONTENT=$(curl -s "$NAVBAR_URL")

# Check if the curl command was successful
if [ $? -ne 0 ]; then
    echo "Failed to fetch the navbar content. Please check the URL and your internet connection."
    exit 1
fi

# Function to update navbar in a file
update_navbar() {
    local file="$1"
    
    # Remove old navbar if it exists
    sed -i '/<!-- NAVBAR START -->/,/<!-- NAVBAR END -->/d' "$file"
    
    # Escape the navbar content to avoid interpretation issues in sed
    escaped_content=$(printf '%s\n' "$NAVBAR_CONTENT" | sed -e 's/[]\/$*.^[]/\\&/g')
    
    # Add new navbar after <body> tag
    sed -i "/<body>/a $escaped_content" "$file"
    
    echo "Updated navbar in $file"
}


# Find all HTML files in the current directory and subdirectories
find . -type f -name "*.html" | while read -r file; do
    update_navbar "$file"
done

echo "Navbar update complete."