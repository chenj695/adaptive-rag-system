# Push Instructions

Your local repo is ahead of GitHub by 2 commits with the demo assets.

## To Push to GitHub:

### Option 1: Using GitHub Desktop or CLI with fresh token
1. Create a new token at https://github.com/settings/tokens
2. Run:
   ```bash
   cd /home/engine/project/rag_system
   git remote set-url origin https://YOUR_NEW_TOKEN@github.com/chenj695/adaptive-rag-system.git
   git push origin main
   ```

### Option 2: Manual Upload
1. Go to https://github.com/chenj695/adaptive-rag-system
2. Upload the files manually via the web interface:
   - `demo_screenshot.png` (381KB)
   - `demo_video.webm` (2.0MB)
   - Updated `README.md`

## What Changed

1. **README.md** - Updated with:
   - Demo screenshot and video
   - Fixed Features section (removed "Multiple Answer Types", added RAPTOR & Evaluation)
   - Changed "20 PDFs" to "PDFs"
   - Added atmospheric science theme description

2. **demo_screenshot.png** - UI screenshot showing weather-themed interface

3. **demo_video.webm** - 4-second video demo with scrolling
