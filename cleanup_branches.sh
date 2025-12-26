#!/bin/bash

# =============================================================================
# Clean Git Branch Structure
# =============================================================================
# Current situation: Too many branches, all pointing to same commit
# Goal: Keep only 'main' (current work) and 'debug' (playground)
# =============================================================================

set -e

echo "========================================"
echo "Cleaning Git Branch Structure"
echo "========================================"
echo ""

# Check current working directory
if [ ! -f "scripts/explore.py" ] && [ ! -f "nbh/explore.py" ]; then
    echo "ERROR: Not in NBH repository root!"
    exit 1
fi

# Show current branches
echo "Current branches:"
git branch
echo ""

# Get current branch
CURRENT=$(git branch --show-current)
echo "You are on: $CURRENT"
echo ""

# =============================================================================
# Step 1: Switch to main branch (ensure it has latest work)
# =============================================================================

echo "Step 1: Switching to main branch..."
git checkout main

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "You have uncommitted changes. Stashing them..."
    git stash save "Auto-stash before branch cleanup"
    STASHED=true
else
    STASHED=false
fi

echo "✓ On main branch"
echo ""

# =============================================================================
# Step 2: Create debug branch from main
# =============================================================================

echo "Step 2: Creating 'debug' branch as playground..."

# Delete debug if it already exists
git branch -D debug 2>/dev/null || true

# Create fresh debug branch from main
git checkout -b debug
echo "✓ Created 'debug' branch from main"
echo ""

# Switch back to main
git checkout main
echo "✓ Back on main"
echo ""

# =============================================================================
# Step 3: Delete unnecessary branches
# =============================================================================

echo "Step 3: Deleting unnecessary branches..."

# List of branches to delete
BRANCHES_TO_DELETE=(
    "main-backup"
    "main-backup-20251226-163936"
    "Graph_only"
    "develop"
    "feature/conservative-nbh"
    "feature/aggressive-nbh"
)

for branch in "${BRANCHES_TO_DELETE[@]}"; do
    if git show-ref --verify --quiet "refs/heads/$branch"; then
        echo "  Deleting: $branch"
        git branch -D "$branch" 2>/dev/null || echo "    (already deleted or protected)"
    fi
done

echo "✓ Cleaned up old branches"
echo ""

# =============================================================================
# Step 4: Restore stashed changes (if any)
# =============================================================================

if [ "$STASHED" = true ]; then
    echo "Step 4: Restoring your uncommitted changes..."
    git stash pop
    echo "✓ Changes restored"
    echo ""
fi

# =============================================================================
# Step 5: Show final state
# =============================================================================

echo "========================================"
echo "✓ Cleanup Complete!"
echo "========================================"
echo ""
echo "Final branch structure:"
git branch -v
echo ""

cat << 'EOF'
========================================
Your New Simple Workflow:
========================================

Two branches:
  • main  - Your stable, working code
  • debug - Playground for experiments

Usage:
------

1. Work on stable features:
   $ git checkout main
   $ # Make changes...
   $ git add .
   $ git commit -m "Add feature X"

2. Try experiments in playground:
   $ git checkout debug
   $ # Experiment freely...
   $ git commit -m "Experiment: trying uncertainty variant"

3. Merge successful experiments to main:
   $ git checkout main
   $ git merge debug
   # Or cherry-pick specific commits:
   $ git cherry-pick <commit-hash>

4. Reset debug if experiments failed:
   $ git checkout debug
   $ git reset --hard main  # Start fresh from main

========================================
Next Steps for Ablation Study:
========================================

When ready to implement conservative/aggressive variants:

Option A - Simple (use debug branch):
  $ git checkout debug
  $ # Edit configs/base.yaml to add uncertainty_mode
  $ # Run experiments, compare results

Option B - Organized (create variant branches later):
  $ git checkout -b experiment/conservative main
  $ git checkout -b experiment/aggressive main
  $ # Develop each variant separately

For now, stick with main + debug!
========================================
EOF

echo ""
echo "Current branch: $(git branch --show-current)"
