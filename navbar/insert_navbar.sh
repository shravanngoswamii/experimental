#!/bin/bash
set -euo pipefail

# Print usage and exit.
usage() {
  echo "Usage: $0 <html-directory> <navbar-url-or-file> [--exclude <path1,path2,...>]"
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

HTML_DIR="$1"
NAVBAR_SOURCE="$2"
shift 2

# Process optional arguments.
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

# Fetch navbar HTML from a URL or local file.
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

# Write the navbar HTML into a temporary file (if you prefer to use sed’s r command).
TMP_NAVBAR=$(mktemp)
echo "$NAVBAR_HTML" > "$TMP_NAVBAR"

# Function to decide if a file should be skipped.
should_exclude() {
  local file="$1"
  IFS=',' read -ra EXCLUDES <<< "$EXCLUDE_LIST"
  for excl in "${EXCLUDES[@]}"; do
    if [[ "$file" == *"$excl"* ]]; then
      return 0  # file should be excluded
    fi
  done
  return 1  # do not exclude
}

# Process each HTML file recursively.
# We use GNU sed’s -z to treat the entire file as a single string.
find "$HTML_DIR" -type f -name "*.html" | while read -r file; do
  if [ -n "$EXCLUDE_LIST" ] && should_exclude "$file"; then
    echo "Skipping excluded file: $file"
    continue
  fi

  echo "Processing $file"

  # (a) Remove any existing navbar block.
  # We match from <!-- NAVBAR START --> to <!-- NAVBAR END --> even if newlines exist.
  sed -z -i.bak -E 's/<!--[[:space:]]*NAVBAR START[[:space:]]*-->.*<!--[[:space:]]*NAVBAR END[[:space:]]*-->//I' "$file"

  # (b) Insert the new navbar immediately after the opening <body> tag.
  # We use a substitution that finds (<body ...>) and replaces it with itself followed by a newline,
  # then the navbar HTML, then another newline.
  #
  # To be safe, we escape any sed‑special characters in the navbar content.
  NAVBAR_ESCAPED=$(printf '%s\n' "$NAVBAR_HTML" | sed 's/[\/&]/\\&/g')
  sed -z -i.bak -E "s/(<body[^>]*>)/\1\
${NAVBAR_ESCAPED}\
/I" "$file"

  # Remove the backup file.
  rm -f "$file.bak"
done

rm -f "$TMP_NAVBAR"

echo "Navbar update complete."
