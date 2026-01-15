#!/usr/bin/env bash
# CI diagnostic tool for GitHub Actions
# Usage: ./scripts/ci-diagnose.sh [status|diagnose] [run_id]

set -euo pipefail

# Colors (using $'...' for escape sequence interpretation)
YELLOW=$'\033[1;33m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'

# Check dependencies
if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI is required. Install with: brew install gh" >&2
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: brew install jq" >&2
    exit 1
fi

# Disable pager for gh commands
export GH_PAGER=""

# Get run info (uses provided run_id or fetches latest)
get_run_info() {
    local run_id_arg="${1:-}"

    if [[ -n "$run_id_arg" ]]; then
        RUN_ID="$run_id_arg"
    else
        RUN_ID=$(gh run list --limit 1 --json databaseId --jq '.[0].databaseId')
    fi

    REPO=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')
    # Fetch run metadata in a single API call
    read -r BRANCH STATUS COMMIT STARTED_AT < <(gh run view "$RUN_ID" --json headBranch,status,headSha,createdAt --jq '[.headBranch, .status, .headSha[0:7], .createdAt] | @tsv')
    RUN_URL="https://github.com/$REPO/actions/runs/$RUN_ID"
}

# Display run header
show_header() {
    # Format timestamp to user locale
    local started_fmt
    started_fmt=$(date -d "$STARTED_AT" 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "$STARTED_AT" 2>/dev/null || echo "$STARTED_AT")

    echo "📊 CI Run $RUN_ID ($BRANCH)"
    printf '%b🔗 %s%b\n' "$DIM" "$RUN_URL" "$RESET"
    printf '%b   commit: %s | started: %s%b\n' "$DIM" "$COMMIT" "$started_fmt" "$RESET"
    echo ""
}

# Show CI status overview
cmd_status() {
    get_run_info "$1"
    show_header

    # Fetch job data once
    JOBS_DATA=$(gh run view "$RUN_ID" --json jobs --jq '.jobs')

    echo "$JOBS_DATA" | jq -r '
        [.[] | select(.name | startswith("Test / Python"))] |
        {
            p: ([.[] | select(.conclusion == "success")] | length),
            f: ([.[] | select(.conclusion == "failure")] | length),
            c: ([.[] | select(.conclusion == "cancelled")] | length),
            r: ([.[] | select(.status == "in_progress")] | length)
        } | "✅ \(.p) passed | ❌ \(.f) failed | 🚫 \(.c) cancelled | 🔄 \(.r) running"'
    echo ""
    echo "$JOBS_DATA" | jq -r '
        .[]
        | select(.name | startswith("Test / Python"))
        | (if .conclusion == "success" then "✅"
           elif .conclusion == "failure" then "❌"
           elif .conclusion == "cancelled" then "🚫"
           elif .status == "in_progress" then "🔄"
           else "⏳" end) + " " + .name'
}

# Filter out warnings and noise from pytest output
filter_noise() {
    grep -v -E \
        -e "DeprecationWarning:" \
        -e "warnings summary" \
        -e "warnings$" \
        -e "pytest-of-runner" \
        -e "site-packages/" \
        -e "^[[:space:]]*$" \
        -e "-- Docs: https://docs.pytest.org" \
        -e "datetime.datetime.utcnow" \
        -e "asyncio.get_event_loop_policy" \
        -e "asyncio.set_event_loop_policy" \
        -e "asyncio.iscoroutinefunction" \
        -e "slated for removal" \
        -e "^tests/.*warnings$" \
        -e "^[[:space:]]+/.*\.py:[0-9]+:" \
    || true
}

# Format pytest failures with colors
# Yellow for test names only
format_failures() {
    awk -v yellow="$YELLOW" -v reset="$RESET" '
        /^=+$/ || /^=+ .* =+$/ { next }
        /^\[gw[0-9]+\]/ { next }
        /^_+ .* _+$/ {
            gsub(/^_+ /, ""); gsub(/ _+$/, "");
            printf "\n%s❌ %s%s\n", yellow, $0, reset
            next
        }
        /^E   / {
            gsub(/^E   /, "");
            printf "   → %s\n", $0
            next
        }
        /./ {
            gsub(/^[ ]+/, "")
            printf "   %s\n", $0
        }
    '
}

# Diagnose CI failures
cmd_diagnose() {
    get_run_info "$1"
    show_header

    # Fetch all job data once and cache it
    JOBS_DATA=$(gh run view "$RUN_ID" --json jobs --jq '.jobs')

    # Show summary counts
    echo "$JOBS_DATA" | jq -r '
        {
            p: ([.[] | select(.conclusion == "success")] | length),
            f: ([.[] | select(.conclusion == "failure")] | length),
            c: ([.[] | select(.conclusion == "cancelled")] | length),
            r: ([.[] | select(.status == "in_progress")] | length)
        } | "✅ \(.p) passed | ❌ \(.f) failed | 🚫 \(.c) cancelled | 🔄 \(.r) running"'
    echo ""

    # Get failed job IDs (space-separated for bash iteration)
    FAILED_JOBS=$(echo "$JOBS_DATA" | jq -r '[.[] | select(.conclusion=="failure") | .databaseId] | join(" ")')

    if [[ -z "$FAILED_JOBS" ]]; then
        if [[ "$STATUS" == "completed" ]]; then
            echo "✅ All jobs passed!"
        else
            echo "🔄 No failures yet (run still in progress)"
        fi
        exit 0
    fi

    echo "❌ Failed Jobs:"
    echo "$JOBS_DATA" | jq -r '.[] | select(.conclusion=="failure") | "  • \(.name)"'
    echo ""

    # Collect all errors for summary
    ALL_ERRORS_FILE=$(mktemp)
    trap 'rm -f "$ALL_ERRORS_FILE"' EXIT

    # Process each failed job
    for JOB_ID in $FAILED_JOBS; do
        JOB_NAME=$(echo "$JOBS_DATA" | jq -r ".[] | select(.databaseId==$JOB_ID) | .name")

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printf '%b📋 %s%b\n' "$BOLD" "$JOB_NAME" "$RESET"

        # Fetch logs
        LOGFILE=$(mktemp)
        if ! gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" 2>/dev/null > "$LOGFILE"; then
            echo "  ⚠️  Could not fetch logs for this job"
            rm -f "$LOGFILE"
            echo ""
            continue
        fi

        # Extract and display failures (filtered)
        # Note: || true handles empty output (e.g., Rust jobs without pytest format)
        sed -n '/= FAILURES =/,/= short test summary/p' "$LOGFILE" | \
            sed 's/^[0-9T:.Z-]* //' | \
            grep -v "^= short test summary" | \
            filter_noise | \
            format_failures | \
            head -500 || true

        # Collect errors for summary (from short test summary section)
        # Extract just the core error message from FAILED lines for clean grouping
        sed -n '/short test summary/,/passed.*failed/p' "$LOGFILE" | \
            sed 's/^[0-9T:.Z-]* //' | \
            grep '^FAILED ' | \
            sed 's/FAILED [^ ]* - //' >> "$ALL_ERRORS_FILE" || true

        rm -f "$LOGFILE"
        echo ""
    done

    # Show error summary grouped by type
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf '%b📊 Error Summary (grouped by type)%b\n' "$BOLD" "$RESET"
    echo ""

    if [[ -s "$ALL_ERRORS_FILE" ]]; then
        sort "$ALL_ERRORS_FILE" | uniq -c | sort -rn | while read -r count error; do
            printf '%b  %3d×%b %s\n' "$YELLOW" "$count" "$RESET" "$error"
        done
    else
        echo "  No error details extracted"
    fi
    echo ""
}

# Main
RUN_ID_ARG="${2:-}"

case "${1:-diagnose}" in
    status)
        cmd_status "$RUN_ID_ARG"
        ;;
    diagnose)
        cmd_diagnose "$RUN_ID_ARG"
        ;;
    *)
        echo "Usage: $0 [status|diagnose] [run_id]"
        exit 1
        ;;
esac
