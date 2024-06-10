#!/bin/bash

# Directory containing HTML files (passed as the first argument to the script)
HTML_DIR=$1

# Generate the navbar HTML
./generate_navbar.sh

# Read the generated navbar HTML
NAVBAR_HTML=$(<navbar.html)

# Process each HTML file in the directory
for file in $(find $HTML_DIR -name "*.html"); do
    # Read the contents of the HTML file
    file_contents=$(<"$file")

    # Insert the navbar HTML after the <body> tag
    updated_contents="${file_contents/<body>/<body>$NAVBAR_HTML}"

    # Write the updated contents back to the file
    echo "$updated_contents" > "$file"
    echo "Updated $file"
done
