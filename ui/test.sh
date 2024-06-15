#!/bin/bash
# Function to add deprecation warning and remove /v0.31/
add_deprecation_warning() {
    local file="$1"

    # Backup the original file
    cp "$file" "$file.bak"

    # Use sed to insert the deprecation warning just after <body> tag
    sed -i '/<body dir="ltr" data-md-color-primary="red" data-md-color-accent="red">/a\
    <style>\
        .deprecated {\
            position: fixed;\
            top: 0;\
            left: 0;\
            right: 0;\
            background-color: rgb(204, 138, 151);\
            color: white;\
            padding: 10px;\
            text-align: center;\
        }\
        .deprecated a {\
            color: rgb(19, 7, 191);\
            text-decoration: underline;\
        }\
    </style>\
    <div class="deprecated">\
        This website is deprecated. Please visit our new website <a href="https://turinglang.org/docs">here</a>.\
    </div>' "$file"

    # Remove /v0.31/
    sed -i 's/\/v0\.31\///g' "$file"

    echo "Deprecation warning added and /v0.31/ removed in $file"
}

# Function to recursively process directories
process_directory() {
    local dir="$1"

    # Process each file in the current directory
    for file in "$dir"/*.html; do
        if [ -f "$file" ]; then
            add_deprecation_warning "$file"
        fi
    done

    # Recursively process subdirectories
    for subdir in "$dir"/*/; do
        if [ -d "$subdir" ]; then
            process_directory "$subdir"
        fi
    done
}

# Main execution starts here
# Process current directory
process_directory "."

# Delete temporary backup files (*.html.bak)
find . -type f -name '*.html.bak' -delete
echo "All HTML files processed and temporary backup files deleted."
