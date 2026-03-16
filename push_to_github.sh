#!/bin/bash
# Push script for adaptive-rag-system
# Usage: ./push_to_github.sh YOUR_GITHUB_TOKEN

if [ -z "$1" ]; then
    echo "Usage: ./push_to_github.sh YOUR_GITHUB_TOKEN"
    echo ""
    echo "To create a token:"
    echo "1. Go to https://github.com/settings/tokens"
    echo "2. Click 'Generate new token (classic)'"
    echo "3. Select 'repo' scope"
    echo "4. Copy the token and paste it here"
    exit 1
fi

TOKEN=$1
REPO_URL="https://${TOKEN}@github.com/chenj695/adaptive-rag-system.git"

echo "Pushing to GitHub..."
cd /home/engine/project/rag_system
git remote set-url origin "$REPO_URL"
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to https://github.com/chenj695/adaptive-rag-system"
    echo ""
    # Reset remote to HTTPS for security
    git remote set-url origin https://github.com/chenj695/adaptive-rag-system.git
else
    echo ""
    echo "❌ Push failed. Check your token and try again."
fi
