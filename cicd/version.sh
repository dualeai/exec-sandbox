#!/bin/bash

# Parse arguments
GIT_DIR="."
COMMIT_HASH=false
METADATA=false

while getopts "g:cm" opt; do
    case $opt in
        g) GIT_DIR="$OPTARG" ;;
        c) COMMIT_HASH=true ;;
        m) METADATA=true ;;
        *) exit 1 ;;
    esac
done

# Get version from git tag or default
cd "$GIT_DIR" || exit 1

# Try to get version from git tag
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "0.1.0")

# Remove 'v' prefix if present
VERSION=${VERSION#v}

# Get commit hash if requested
if [ "$COMMIT_HASH" = true ]; then
    COMMIT=$(git rev-parse --short HEAD)
    VERSION="$VERSION-$COMMIT"
fi

# Add metadata if requested
if [ "$METADATA" = true ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    VERSION="$VERSION+$BRANCH.$TIMESTAMP"
fi

echo "$VERSION"
