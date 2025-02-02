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

# Write the navbar HTML into a temporary file.
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

# Define a newline variable.
newline=$'\n'

# Escape characters that might conflict with our delimiter (#) or sed’s replacement (&).
# (Since we use '#' as the delimiter below, we need to escape any literal '#' and '&'.)
NAVBAR_ESCAPED=$(printf '%s' "$NAVBAR_HTML" | sed 's/[#&]/\\&/g')

# Process each HTML file recursively.
# Use GNU sed’s -z so that the entire file is treated as a single string.
find "$HTML_DIR" -type f -name "*.html" | while read -r file; do
  if [ -n "$EXCLUDE_LIST" ] && should_exclude "$file"; then
    echo "Skipping excluded file: $file"
    continue
  fi

  echo "Processing $file"

  # (a) Remove any existing navbar block (from <!-- NAVBAR START --> to <!-- NAVBAR END -->)
  sed -z -i.bak -E 's/<!--[[:space:]]*NAVBAR START[[:space:]]*-->.*<!--[[:space:]]*NAVBAR END[[:space:]]*-->//I' "$file"

  # (b) Insert the new navbar immediately after the opening <body> tag.
  # We use '#' as the delimiter so we don’t have to worry about '/' in the navbar HTML.
  sed -z -i.bak -E "s#(<body[^>]*>)#\1${newline}${NAVBAR_ESCAPED}${newline}#I" "$file"

  # Remove the backup file.
  rm -f "$file.bak"
done

rm -f "$TMP_NAVBAR"

echo "Navbar update complete."
