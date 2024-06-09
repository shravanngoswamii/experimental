#!/bin/bash

# Define the HTML content to be added
html_to_add=$(curl -s https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar.html)

# Loop through each HTML file in the current directory
for file in *.html; do
    # Check if the file exists and is not the navbar.html file itself
    if [ -f "$file" ] && [ "$file" != "navbar.html" ]; then
        # Temporarily store the content of the original file
        original_content=$(cat "$file")
        
        # Find the position of the opening <body> tag
        body_tag_position=$(awk '/<body/{ print NR; exit }' "$file")
        
        # Extract content before and after the body tag
        content_before_body=$(awk "NR==1, NR==$body_tag_position" "$file")
        content_after_body=$(awk "NR==$body_tag_position, NR==$(wc -l < "$file")" "$file")
        
        # Combine original content with added HTML
        new_content="$content_before_body\n$html_to_add\n$content_after_body"
        
        # Write new content back to the file
        echo -e "$new_content" > "$file"
        
        echo "Added navbar to $file"
    fi
done
