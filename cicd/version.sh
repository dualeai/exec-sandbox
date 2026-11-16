#!/bin/bash
# Print the project version: the nearest tag reachable from HEAD (git describe)
# without its "v" prefix, or 0.0.0.dev0 when there is none.
#
# Usage: cicd/version.sh [-g <repo dir>] [-c] [-m]
#   -g  repository directory (default: .)
#   -c  append "-<short commit hash>"
#   -m  append "+<branch>.<local time as YYYYMMDDHHMMSS>"

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

# Run git in the target repository
cd "$GIT_DIR" || exit 1

# Try to get version from git tag
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "0.0.0.dev0")

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
