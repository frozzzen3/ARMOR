#!/bin/bash

# Script to rename all 'renders' directories to 'render_gs'
# Usage: ./rename_gs_dir.sh <root_directory>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <root_directory>"
    exit 1
fi

ROOT_DIR="$1"

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' does not exist"
    exit 1
fi

# Find all directories named 'renders' and rename them to 'render_gs'
find "$ROOT_DIR" -type d -name "renders" | while read -r dir; do
    parent_dir=$(dirname "$dir")
    new_dir="$parent_dir/renders_gs"
    
    if [ -d "$new_dir" ]; then
        echo "Warning: '$new_dir' already exists, skipping '$dir'"
    else
        mv "$dir" "$new_dir"
        echo "Renamed: $dir → $new_dir"
    fi
done

echo "Done!"