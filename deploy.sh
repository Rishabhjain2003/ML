#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Run this on your LOCAL Mac to:
#   1. Upload the dataset zip to the cloud instance
#   2. Clone the GitHub repo on the instance
#   3. Run setup.sh remotely to install deps + launch the pipeline
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh <username> [zip_path] [github_repo_url]
#
# Examples:
#   ./deploy.sh john
#   ./deploy.sh john ~/Downloads/6ab10eb3b23ba_student_resource.zip
#   ./deploy.sh john ~/Downloads/6ab10eb3b23ba_student_resource.zip https://github.com/rishabhjain/entity-resolution.git
# =============================================================================

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
USERNAME="${1:-}"
CLOUD_IP="130.141.174.109"
DATASET_ZIP="${2:-$HOME/Downloads/6ab10eb3b23ba_student_resource.zip}"
GITHUB_REPO="${3:-}"   # e.g. https://github.com/yourname/entity-resolution.git

if [[ -z "$USERNAME" ]]; then
    echo "Usage: ./deploy.sh <username> [zip_path] [github_repo_url]"
    echo "  username     — SSH username for $CLOUD_IP"
    echo "  zip_path     — path to dataset zip (default: ~/Downloads/6ab10eb3b23ba_student_resource.zip)"
    echo "  github_repo  — GitHub repo URL to clone (optional if already cloned)"
    exit 1
fi

SSH_HOST="$USERNAME@$CLOUD_IP"
REMOTE_HOME="/home/$USERNAME"
REMOTE_PROJECT="$REMOTE_HOME/MLChallenge"

echo "============================================================"
echo "  Deploying Entity Resolution Pipeline"
echo "  Target : $SSH_HOST"
echo "  Dataset: $DATASET_ZIP"
echo "============================================================"

# ── Step 1: Verify dataset zip exists locally ─────────────────────────────────
if [[ ! -f "$DATASET_ZIP" ]]; then
    echo "[ERROR] Dataset zip not found at: $DATASET_ZIP"
    echo "  Pass the correct path as the second argument."
    exit 1
fi
echo "[1/5] Dataset zip found: $(du -sh "$DATASET_ZIP" | cut -f1)"

# ── Step 2: Upload dataset zip to cloud instance ──────────────────────────────
echo "[2/5] Uploading dataset zip to $SSH_HOST …"
scp -o StrictHostKeyChecking=no \
    "$DATASET_ZIP" \
    "$SSH_HOST:$REMOTE_HOME/dataset.zip"
echo "      Upload complete."

# ── Step 3: Upload setup.sh to cloud instance ─────────────────────────────────
echo "[3/5] Uploading setup.sh …"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scp -o StrictHostKeyChecking=no \
    "$SCRIPT_DIR/setup.sh" \
    "$SSH_HOST:$REMOTE_HOME/setup.sh"

# ── Step 4: Run setup.sh on the cloud instance ────────────────────────────────
echo "[4/5] Running setup on cloud instance …"
ssh -o StrictHostKeyChecking=no "$SSH_HOST" bash << REMOTE_SCRIPT
    chmod +x ~/setup.sh
    ~/setup.sh "$GITHUB_REPO"
REMOTE_SCRIPT

# ── Step 5: Done ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Deployment complete."
echo ""
echo "  Monitor progress:"
echo "    ssh $SSH_HOST"
echo "    tail -f ~/MLChallenge/logs/pipeline.log"
echo "    screen -r mlpipeline"
echo ""
echo "  Check process:"
echo "    ssh $SSH_HOST 'ps aux | grep pipeline.py'"
