#!/usr/bin/env bash
# CI diagnostic tool for GitHub Actions
# Usage: ./scripts/ci-diagnose.sh [status|diagnose] [run_id]

set -euo pipefail

# Colors
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Check dependencies
if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI is required. Install with: brew install gh" >&2
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
    BRANCH=$(gh run view "$RUN_ID" --json headBranch --jq '.headBranch')
    STATUS=$(gh run view "$RUN_ID" --json status --jq '.status')
    RUN_URL="https://github.com/$REPO/actions/runs/$RUN_ID"
}

# Show CI status overview
cmd_status() {
    get_run_info "$1"

    echo "📊 CI Run $RUN_ID ($BRANCH)"
    printf '%b🔗 %s%b\n' "$DIM" "$RUN_URL" "$RESET"
    echo ""
    gh run view "$RUN_ID" --json jobs --jq '
        {
            p: ([.jobs[] | select(.name | startswith("Test / Python")) | select(.conclusion == "success")] | length),
            f: ([.jobs[] | select(.name | startswith("Test / Python")) | select(.conclusion == "failure")] | length),
            r: ([.jobs[] | select(.name | startswith("Test / Python")) | select(.status == "in_progress")] | length)
        } | "✅ \(.p) passed | ❌ \(.f) failed | 🔄 \(.r) running"'
    echo ""
    gh run view "$RUN_ID" --json jobs --jq '
        .jobs[]
        | select(.name | startswith("Test / Python"))
        | (if .conclusion == "success" then "✅"
           elif .conclusion == "failure" then "❌"
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
format_failures() {
    awk '
        /^={10,}/ { next }
        /^_{4,}.*_{4,}$/ {
            gsub(/^_+ /, ""); gsub(/ _+$/, "");
            printf "\n\033[1;31m❌ %s\033[0m\n", $0
            next
        }
        /^\[gw[0-9]+\]/ {
            printf "\033[2m   %s\033[0m\n", $0
            next
        }
        /^E   / {
            gsub(/^E   /, "");
            printf "\033[1;33m   → %s\033[0m\n", $0
            next
        }
        /^-{10,}.*-{10,}$/ {
            printf "\033[2m   %s\033[0m\n", $0
            next
        }
        /^[A-Z]+  / {
            printf "\033[2m   %s\033[0m\n", $0
            next
        }
        /^[a-z_\/]+\.py:[0-9]+:/ {
            printf "   \033[36m%s\033[0m\n", $0
            next
        }
        /^    / {
            printf "   %s\n", $0
            next
        }
        /./ { print "   " $0 }
    '
}

# Diagnose CI failures
cmd_diagnose() {
    get_run_info "$1"

    echo "📊 CI Run $RUN_ID ($BRANCH)"
    printf '%b🔗 %s%b\n' "$DIM" "$RUN_URL" "$RESET"
    echo ""
    gh run view "$RUN_ID" --json jobs --jq '
        {
            p: ([.jobs[] | select(.conclusion == "success")] | length),
            f: ([.jobs[] | select(.conclusion == "failure")] | length),
            r: ([.jobs[] | select(.status == "in_progress")] | length)
        } | "✅ \(.p) passed | ❌ \(.f) failed | 🔄 \(.r) running"'
    echo ""

    # Get failed job IDs (space-separated for bash iteration)
    FAILED_JOBS=$(gh run view "$RUN_ID" --json jobs --jq '[.jobs[] | select(.conclusion=="failure") | .databaseId] | join(" ")')

    if [[ -z "$FAILED_JOBS" ]]; then
        if [[ "$STATUS" == "completed" ]]; then
            echo "✅ All jobs passed!"
        else
            echo "🔄 No failures yet (run still in progress)"
        fi
        exit 0
    fi

    echo "❌ Failed Jobs:"
    gh run view "$RUN_ID" --json jobs --jq '.jobs[] | select(.conclusion=="failure") | "  • \(.name)"'
    echo ""

    # Collect all errors for summary
    ALL_ERRORS_FILE=$(mktemp)
    trap 'rm -f "$ALL_ERRORS_FILE"' EXIT

    # Process each failed job
    for JOB_ID in $FAILED_JOBS; do
        JOB_NAME=$(gh run view "$RUN_ID" --json jobs --jq ".jobs[] | select(.databaseId==$JOB_ID) | .name")

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printf '%b📋 %s%b\n' "$BOLD" "$JOB_NAME" "$RESET"

        # Fetch logs
        LOGFILE=$(mktemp)
        gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" 2>/dev/null > "$LOGFILE"

        # Extract and display failures (filtered)
        # Note: || true handles empty output (e.g., Rust jobs without pytest format)
        sed -n '/= FAILURES =/,/= short test summary/p' "$LOGFILE" | \
            sed 's/^[0-9T:.Z-]* //' | \
            grep -v "^= short test summary" | \
            filter_noise | \
            format_failures | \
            head -500 || true

        # Collect errors for summary (from short test summary section)
        sed -n '/short test summary/,/passed.*failed/p' "$LOGFILE" | \
            grep "FAILED " | \
            sed 's/^[0-9T:.Z-]* //' | \
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
