#!/bin/bash
set -euo pipefail

# Usage message
usage() {
  echo "Usage: $0 <html-directory> <navbar-url-or-file> [--exclude <path1,path2,...>]"
  exit 1
}

# Check arguments
if [ "$#" -lt 2 ]; then
  usage
fi

HTML_DIR="$1"
NAVBAR_SOURCE="$2"
shift 2

# Process optional arguments
EXCLUDE_LIST=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --exclude)
      EXCLUDE_LIST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Get navbar HTML from URL or file
if [[ "$NAVBAR_SOURCE" =~ ^https?:// ]]; then
  NAVBAR_HTML=$(curl -s "$NAVBAR_SOURCE")
else
  if [ -f "$NAVBAR_SOURCE" ]; then
    NAVBAR_HTML=$(cat "$NAVBAR_SOURCE")
  else
    echo "Error: File '$NAVBAR_SOURCE' not found."
    exit 1
  fi
fi

if [ -z "$NAVBAR_HTML" ]; then
  echo "Error: Navbar HTML is empty."
  exit 1
fi

# Create a temporary file that will hold the navbar content.
# (It should already include the comment markers.)
TMP_NAVBAR=$(mktemp)
echo "$NAVBAR_HTML" > "$TMP_NAVBAR"

# Optional: function to decide whether a file should be excluded.
should_exclude() {
  local file="$1"
  IFS=',' read -ra EXCLUDES <<< "$EXCLUDE_LIST"
  for excl in "${EXCLUDES[@]}"; do
    if [[ "$file" == *"$excl"* ]]; then
      return 0  # exclude this file
    fi
  done
  return 1  # do not exclude
}

# Process each HTML file recursively
find "$HTML_DIR" -type f -name "*.html" | while read -r file; do

  if [ -n "$EXCLUDE_LIST" ] && should_exclude "$file"; then
    echo "Skipping excluded file: $file"
    continue
  fi

  echo "Processing $file"

  # (a) Remove any existing navbar block.
  # This sed range deletes from the line that contains <!-- NAVBAR START -->
  # up to (and including) the line that contains <!-- NAVBAR END -->.
  sed -i.bak '/<!--[[:space:]]*NAVBAR START[[:space:]]*-->/, /<!--[[:space:]]*NAVBAR END[[:space:]]*-->/d' "$file"

  # (b) Insert the new navbar immediately after the opening <body> tag.
  # This command looks for the first occurrence of <body ...> (case‑insensitive)
  # and uses the "r" (read file) command to insert the contents of the temporary file.
  sed -i.bak -E '/<body[^>]*>/I {
    # The matched line is left unchanged, then the content from TMP_NAVBAR is inserted.
    r '"$TMP_NAVBAR"'
  }' "$file"

  # Remove the backup created by sed (optional)
  rm -f "$file.bak"
done

# Clean up temporary file
rm -f "$TMP_NAVBAR"

echo "Navbar update complete."
