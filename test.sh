#!/bin/bash

# === Config ===
SRC_REPO="https://github.com/TuringLang/TuringBenchmarking.jl.git"
SRC_BRANCH="gh-pages"
SRC_TMP_DIR="tmp_TuringBenchmarking"
DEST_PREFIX="/Deprecated/TuringBenchmarking"
DEST_BASE_DIR="TuringBenchmarking.jl"

# === Clean previous temp clone ===
rm -rf "$SRC_TMP_DIR"
git clone --branch "$SRC_BRANCH" --depth 1 "$SRC_REPO" "$SRC_TMP_DIR"

cd "$SRC_TMP_DIR" || exit 1

# === Match dev, stable, v*, and symlinks ===
MATCHED_DIRS=$(
  find . -maxdepth 1 \( -type d -o -type l \) \
  ! -name "." \
  -exec bash -c '
    for d; do
      base=$(basename "$d")
      [[ "$base" == "dev" || "$base" == "stable" || "$base" == v* ]] && echo "$base"
    done
  ' _ {} +
)

cd ..

# === Create minimal redirect folders ===
mkdir -p "$DEST_BASE_DIR"
for path in $MATCHED_DIRS; do
  dest_dir="${DEST_BASE_DIR}/${path}"
  mkdir -p "$dest_dir"
  cat > "${dest_dir}/index.html" <<EOF
<meta http-equiv="refresh" content="0; url=${DEST_PREFIX}/${path}/" />
EOF
  echo "✅ Created redirect for: $path"
done

# === Create root redirect to stable ===
cat > "${DEST_BASE_DIR}/index.html" <<EOF
<meta http-equiv="refresh" content="0; url=${DEST_PREFIX}/stable/" />
EOF
echo "✅ Created root redirect: /TuringBenchmarking.jl/ → /Deprecated/TuringBenchmarking/stable/"

# === Cleanup ===
rm -rf "$SRC_TMP_DIR"
echo "🎉 All redirects created in ./$DEST_BASE_DIR/"
