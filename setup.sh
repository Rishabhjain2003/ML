#!/usr/bin/env bash
# =============================================================================
# setup.sh — Run this ON the cloud instance to:
#   1. Detect OS and install system dependencies
#   2. Extract the dataset zip into the correct directory structure
#   3. Clone the GitHub repo (or pull latest if already cloned)
#   4. Create Python venv and install all dependencies
#   5. Launch the pipeline in a detached screen session
#
# Called automatically by deploy.sh, or run manually:
#   chmod +x setup.sh
#   ./setup.sh https://github.com/yourname/entity-resolution.git
#   ./setup.sh   # if repo is already cloned
# =============================================================================

set -euo pipefail

GITHUB_REPO="${1:-}"
CLOUD_IP="130.141.174.109"
PROJECT_DIR="$HOME/MLChallenge"
DATASET_ZIP="$HOME/dataset.zip"
LOG_DIR="$PROJECT_DIR/logs"
PYTHON_MIN_VERSION="3.10"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

# =============================================================================
section "1. System check"
# =============================================================================

# Detect OS
if   [[ -f /etc/debian_version ]]; then OS="debian"
elif [[ -f /etc/redhat-release ]]; then OS="redhat"
elif [[ "$(uname)" == "Darwin" ]];  then OS="macos"
else                                     OS="unknown"
fi
info "OS detected: $OS"

# Install screen if missing
if ! command -v screen &>/dev/null; then
    warn "screen not found — installing …"
    if   [[ "$OS" == "debian" ]]; then sudo apt-get install -y screen
    elif [[ "$OS" == "redhat" ]]; then sudo yum install -y screen
    else warn "Cannot auto-install screen on $OS — install it manually"
    fi
fi

# Find best available Python (3.10+)
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 10 ]]; }; then
            PYTHON="$candidate"
            info "Using Python: $(command -v $PYTHON) ($ver)"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    warn "Python 3.10+ not found — attempting install …"
    if [[ "$OS" == "debian" ]]; then
        sudo apt-get update -y
        sudo apt-get install -y python3.11 python3.11-venv python3-pip
        PYTHON="python3.11"
    elif [[ "$OS" == "redhat" ]]; then
        sudo yum install -y python311 python311-pip
        PYTHON="python3.11"
    else
        error "Python 3.10+ required. Install it manually and rerun."
    fi
fi

# Install unzip if missing
if ! command -v unzip &>/dev/null; then
    warn "unzip not found — installing …"
    if   [[ "$OS" == "debian" ]]; then sudo apt-get install -y unzip
    elif [[ "$OS" == "redhat" ]]; then sudo yum install -y unzip
    fi
fi

# =============================================================================
section "2. Clone / update repository"
# =============================================================================

if [[ -d "$PROJECT_DIR/.git" ]]; then
    info "Repo already exists — pulling latest …"
    git -C "$PROJECT_DIR" pull
elif [[ -n "$GITHUB_REPO" ]]; then
    info "Cloning $GITHUB_REPO …"
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
else
    # No git repo — create directory structure and copy local files if present
    warn "No GitHub repo provided and no existing clone found."
    warn "Creating directory structure. You will need to copy code/ manually."
    mkdir -p "$PROJECT_DIR/code/src" "$PROJECT_DIR/utils"
fi

# Ensure required directories exist
mkdir -p "$PROJECT_DIR/dataset/train" \
         "$PROJECT_DIR/dataset/test"  \
         "$PROJECT_DIR/models"        \
         "$PROJECT_DIR/output"        \
         "$PROJECT_DIR/logs"

# =============================================================================
section "3. Extract dataset"
# =============================================================================

if [[ ! -f "$DATASET_ZIP" ]]; then
    error "Dataset zip not found at $DATASET_ZIP. Run deploy.sh from your Mac first."
fi

info "Extracting dataset (this may take a minute) …"

# Extract to a temp directory then move into place
TMPDIR_EXTRACT="$HOME/.dataset_extract_tmp"
rm -rf "$TMPDIR_EXTRACT"
mkdir -p "$TMPDIR_EXTRACT"
unzip -q "$DATASET_ZIP" -d "$TMPDIR_EXTRACT"

# The zip has structure: student_resource/dataset/train/*.tsv etc.
# Map it to: MLChallenge/dataset/train/*.tsv
EXTRACTED_DATASET="$TMPDIR_EXTRACT/student_resource/dataset"

if [[ ! -d "$EXTRACTED_DATASET" ]]; then
    # Try finding dataset directory at any depth
    EXTRACTED_DATASET=$(find "$TMPDIR_EXTRACT" -type d -name "dataset" | head -1)
fi

if [[ -z "$EXTRACTED_DATASET" || ! -d "$EXTRACTED_DATASET" ]]; then
    error "Could not find 'dataset' directory inside zip. Check zip structure."
fi

info "Moving train files …"
find "$EXTRACTED_DATASET/train" -name "*.tsv" -exec mv {} "$PROJECT_DIR/dataset/train/" \; 2>/dev/null || true

info "Moving test files …"
find "$EXTRACTED_DATASET/test"  -name "*.tsv" -exec mv {} "$PROJECT_DIR/dataset/test/"  \; 2>/dev/null || true

# Also grab validate_submission.py if present in zip
EXTRACTED_UTILS="$TMPDIR_EXTRACT/student_resource/utils"
if [[ -d "$EXTRACTED_UTILS" ]]; then
    cp -n "$EXTRACTED_UTILS"/*.py "$PROJECT_DIR/utils/" 2>/dev/null || true
fi

rm -rf "$TMPDIR_EXTRACT"

# Verify key files
REQUIRED_FILES=(
    "$PROJECT_DIR/dataset/train/train_source1.tsv"
    "$PROJECT_DIR/dataset/train/train_source2.tsv"
    "$PROJECT_DIR/dataset/train/train_source3.tsv"
    "$PROJECT_DIR/dataset/train/train_ground_truth.tsv"
    "$PROJECT_DIR/dataset/test/test_source1.tsv"
    "$PROJECT_DIR/dataset/test/test_source2.tsv"
    "$PROJECT_DIR/dataset/test/test_source3.tsv"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        info "  ✓ $(basename $f)  ($(du -sh "$f" | cut -f1))"
    else
        warn "  ✗ MISSING: $f"
        ALL_OK=false
    fi
done
$ALL_OK || error "Some dataset files are missing. Check zip contents."

# =============================================================================
section "4. Python virtual environment + dependencies"
# =============================================================================

VENV="$PROJECT_DIR/venv"

if [[ ! -d "$VENV" ]]; then
    info "Creating virtual environment …"
    "$PYTHON" -m venv "$VENV"
fi

info "Upgrading pip …"
"$VENV/bin/pip" install --quiet --upgrade pip

info "Installing dependencies …"
if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
    "$VENV/bin/pip" install --quiet -r "$PROJECT_DIR/requirements.txt"
else
    # Fallback: install directly
    "$VENV/bin/pip" install --quiet \
        lightgbm==4.7.0 \
        rapidfuzz==3.14.6 \
        jellyfish==1.2.1 \
        pandas==3.0.6 \
        numpy==2.5.3 \
        scikit-learn==1.9.1 \
        tqdm==4.70.1
fi
info "Dependencies installed."

# =============================================================================
section "5. Launch pipeline"
# =============================================================================

# Kill any existing pipeline session
screen -S mlpipeline -X quit 2>/dev/null || true
sleep 1

# Kill any stale pipeline processes
pkill -f "pipeline.py" 2>/dev/null || true
sleep 1

# Verify code exists
if [[ ! -f "$PROJECT_DIR/code/pipeline.py" ]]; then
    error "pipeline.py not found at $PROJECT_DIR/code/pipeline.py
  Make sure the GitHub repo was cloned correctly and contains the code/ directory."
fi

info "Starting pipeline in detached screen session …"
screen -dmS mlpipeline \
    "$VENV/bin/python3" -u "$PROJECT_DIR/code/pipeline.py"

sleep 3

# Verify it started
if screen -list | grep -q "mlpipeline"; then
    PIPELINE_PID=$(pgrep -f "pipeline.py" | head -1)
    info "Pipeline running. Screen: mlpipeline  PID: $PIPELINE_PID"
else
    error "Pipeline failed to start. Check logs: tail -f $LOG_DIR/pipeline.log"
fi

# =============================================================================
section "Setup complete"
# =============================================================================
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Entity Resolution Pipeline is running              │"
echo "  │                                                     │"
echo "  │  Monitor:                                           │"
echo "  │    tail -f ~/MLChallenge/logs/pipeline.log          │"
echo "  │    screen -r mlpipeline   (Ctrl-A D to detach)      │"
echo "  │                                                     │"
echo "  │  Output (when done):                                │"
echo "  │    ~/MLChallenge/output/matching_results.tsv        │"
echo "  │    ~/MLChallenge/output/candidate_pairs.tsv         │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
