#!/bin/bash

# =============================================================================
# Safe Push: Backup all branches to GitHub before cleanup
# =============================================================================

set -e

echo "========================================"
echo "Pushing All Branches to GitHub"
echo "========================================"
echo ""

# Check if we have a remote
if ! git remote get-url origin &>/dev/null; then
    echo "ERROR: No 'origin' remote configured!"
    exit 1
fi

REMOTE_URL=$(git remote get-url origin)
echo "Remote: $REMOTE_URL"
echo ""

# =============================================================================
# Step 1: Commit any uncommitted changes on current branch
# =============================================================================

CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo ""
    echo "You have uncommitted changes:"
    git status --short
    echo ""
    read -p "Commit these changes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Commit message: " COMMIT_MSG
        git add -A
        git commit -m "$COMMIT_MSG"
        echo "✓ Changes committed"
    else
        echo "⚠ Skipping uncommitted changes (they won't be pushed)"
    fi
else
    echo "✓ No uncommitted changes"
fi

echo ""

# =============================================================================
# Step 2: Push all local branches to GitHub
# =============================================================================

echo "Pushing all branches to GitHub..."
echo ""

# Get list of all local branches
BRANCHES=$(git branch --format='%(refname:short)')

for branch in $BRANCHES; do
    echo "Pushing: $branch"
    
    # Push branch (create remote if doesn't exist)
    if git push -u origin "$branch" 2>&1; then
        echo "  ✓ $branch pushed successfully"
    else
        echo "  ⚠ Failed to push $branch (may already be up-to-date)"
    fi
done

echo ""
echo "========================================"
echo "✓ All Branches Backed Up to GitHub!"
echo "========================================"
echo ""

# =============================================================================
# Step 3: Show what was pushed
# =============================================================================

echo "Branches on GitHub:"
git ls-remote --heads origin | awk '{print "  " $2}' | sed 's|refs/heads/||'

echo ""
echo "You can now safely run ./cleanup_branches.sh"
echo ""
