#!/usr/bin/env bash
#
# build_orbslam3.sh — Build Pangolin + ORB-SLAM3 + our mono_video runner.
#
# IDEMPOTENT AND RESUMABLE: every stage checks for its own output artifact and
# skips if present. Safe to re-run after a failure or a dropped connection —
# it will not redo work that already succeeded.
#
# CRITICAL PATH RULE: everything is built under $BUILD_DIR in the Linux home
# dir, NEVER under /mnt/c. The Windows project path contains spaces (which
# break several ORB-SLAM3/Pangolin CMake paths) and /mnt/c I/O under WSL2 is
# very slow.
#
# Usage:
#   bash setup/build_orbslam3.sh                 # build everything missing
#   bash setup/build_orbslam3.sh --only-ext      # rebuild just mono_video (fast)
#   BUILD_DIR=~/somewhere bash setup/build_orbslam3.sh
#
# Every applied source patch is appended to setup/BUILD_NOTES.md.

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$HOME/orbslam3_build}"
PANGOLIN_TAG="${PANGOLIN_TAG:-v0.6}"     # v0.6 is the tag ORB-SLAM3 is known to build against;
                                          # Pangolin master has repeatedly broken it.
JOBS="${JOBS:-2}"                         # 2 keeps memory use sane on WSL2 and produces
                                          # steadier output (fewer silent stalls).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTES="$SCRIPT_DIR/BUILD_NOTES.md"

ONLY_EXT=0
[[ "${1:-}" == "--only-ext" ]] && ONLY_EXT=1

mkdir -p "$BUILD_DIR"

log()  { echo -e "\n\033[1;34m==> $*\033[0m"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
warn() { echo -e "\033[1;33m  ! $*\033[0m"; }
die()  { echo -e "\033[1;31m[FATAL] $*\033[0m" >&2; exit 1; }

note_patch() {
    # note_patch <file> <error> <fix>
    [[ -f "$NOTES" ]] || cat > "$NOTES" <<'EOF'
# BUILD_NOTES.md

Every source patch applied by `setup/build_orbslam3.sh` to make ORB-SLAM3 and
Pangolin compile on modern Ubuntu/GCC, recorded so the build is reproducible.

| File | Error | Fix |
|---|---|---|
EOF
    echo "| \`$1\` | $2 | $3 |" >> "$NOTES"
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: apt dependencies
# ─────────────────────────────────────────────────────────────────────────────
if [[ $ONLY_EXT -eq 0 ]]; then
log "Stage 1/5: apt dependencies"

APT_PKGS=(
    build-essential cmake git pkg-config
    libeigen3-dev libopencv-dev libglew-dev
    libboost-all-dev libssl-dev
    libpython3-dev python3-dev
    libepoxy-dev                       # Pangolin v0.6 wants this on newer Ubuntu
)

MISSING=()
for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" &>/dev/null || MISSING+=("$p")
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    ok "All apt dependencies already installed — skipping."
else
    warn "Installing: ${MISSING[*]}"
    warn "This needs sudo. If you're driving this from a tool that can't type a"
    warn "password, run this line manually first, then re-run the script:"
    warn "  sudo apt-get update && sudo apt-get install -y ${MISSING[*]}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING[@]}"
    ok "apt dependencies installed."
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Pangolin
# ─────────────────────────────────────────────────────────────────────────────
if [[ $ONLY_EXT -eq 0 ]]; then
log "Stage 2/5: Pangolin ($PANGOLIN_TAG)"

PANGOLIN_DIR="$BUILD_DIR/Pangolin"

if [[ -f "$PANGOLIN_DIR/build/libpangolin.so" ]] || \
   [[ -f /usr/local/lib/libpangolin.so ]]; then
    ok "Pangolin already built — skipping."
else
    if [[ ! -d "$PANGOLIN_DIR/.git" ]]; then
        rm -rf "$PANGOLIN_DIR"
        git clone --depth 1 --branch "$PANGOLIN_TAG" \
            https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_DIR"
    fi

    # Known patch: Pangolin v0.6 fails on GCC >= 11 with
    #   error: 'std::numeric_limits' has not been declared
    # because <limits> is no longer transitively included.
    DATALOG="$PANGOLIN_DIR/src/plot/datalog.cpp"
    if [[ -f "$DATALOG" ]] && ! grep -q "#include <limits>" "$DATALOG"; then
        sed -i '1i #include <limits>' "$DATALOG"
        note_patch "Pangolin/src/plot/datalog.cpp" \
            "'numeric_limits' is not a member of 'std' (GCC>=11 no longer transitively includes <limits>)" \
            "Added \`#include <limits>\` at top of file"
        ok "Patched datalog.cpp (<limits> include)"
    fi

    mkdir -p "$PANGOLIN_DIR/build"
    cd "$PANGOLIN_DIR/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
        2>&1 | tee cmake.log
    make -j"$JOBS" 2>&1 | tee build.log
    sudo make install
    sudo ldconfig
    ok "Pangolin built and installed."
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: ORB-SLAM3 clone + patches
# ─────────────────────────────────────────────────────────────────────────────
ORB_DIR="$BUILD_DIR/ORB_SLAM3"

if [[ $ONLY_EXT -eq 0 ]]; then
log "Stage 3/5: ORB-SLAM3 source"

if [[ ! -d "$ORB_DIR/.git" ]]; then
    rm -rf "$ORB_DIR"
    git clone https://github.com/UZ-SLAMlab/ORB_SLAM3.git "$ORB_DIR"
    ok "Cloned ORB-SLAM3."
else
    ok "ORB-SLAM3 already cloned."
fi

# Patch: C++11 -> C++14. ORB-SLAM3 master's CMakeLists still asks for C++11 in
# places, but its own Sophus/g2o headers need C++14 on modern GCC.
if grep -q 'CMAKE_CXX_STANDARD 11' "$ORB_DIR/CMakeLists.txt" 2>/dev/null; then
    sed -i 's/CMAKE_CXX_STANDARD 11/CMAKE_CXX_STANDARD 14/' "$ORB_DIR/CMakeLists.txt"
    note_patch "ORB_SLAM3/CMakeLists.txt" \
        "Compile errors from Sophus/g2o headers requiring C++14 features" \
        "CMAKE_CXX_STANDARD 11 -> 14"
    ok "Patched CMakeLists.txt (C++14)"
fi
if grep -q '\-std=c++11' "$ORB_DIR/CMakeLists.txt" 2>/dev/null; then
    sed -i 's/-std=c++11/-std=c++14/g' "$ORB_DIR/CMakeLists.txt"
    note_patch "ORB_SLAM3/CMakeLists.txt" \
        "-std=c++11 conflicts with C++14-requiring headers" \
        "-std=c++11 -> -std=c++14"
    ok "Patched CMakeLists.txt (-std=c++14)"
fi

# Patch: missing <thread>/<chrono>/<unistd.h> in several headers on newer GCC.
for f in "$ORB_DIR/include/System.h" \
         "$ORB_DIR/include/LoopClosing.h" \
         "$ORB_DIR/include/LocalMapping.h" \
         "$ORB_DIR/include/Tracking.h" \
         "$ORB_DIR/include/Viewer.h"; do
    [[ -f "$f" ]] || continue
    if ! grep -q "#include <thread>" "$f"; then
        sed -i '1i #include <thread>\n#include <chrono>\n#include <unistd.h>' "$f"
        note_patch "ORB_SLAM3/$(basename "$f")" \
            "'thread'/'chrono'/'usleep' not declared (GCC>=11 dropped transitive includes)" \
            "Added \`#include <thread>\`, \`<chrono>\`, \`<unistd.h>\`"
    fi
done
ok "Header include patches applied (where needed)."

# Patch: OpenCV 4 removed the CV_ prefixed constants used in a few spots.
if [[ -f "$ORB_DIR/src/Tracking.cc" ]]; then
    sed -i 's/CV_LOAD_IMAGE_UNCHANGED/cv::IMREAD_UNCHANGED/g;
            s/CV_LOAD_IMAGE_GRAYSCALE/cv::IMREAD_GRAYSCALE/g;
            s/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g;
            s/\bCV_BGR2GRAY\b/cv::COLOR_BGR2GRAY/g;
            s/\bCV_RGB2GRAY\b/cv::COLOR_RGB2GRAY/g;
            s/\bCV_BGRA2GRAY\b/cv::COLOR_BGRA2GRAY/g;
            s/\bCV_RGBA2GRAY\b/cv::COLOR_RGBA2GRAY/g' \
        "$ORB_DIR"/src/*.cc 2>/dev/null || true
    ok "OpenCV 4 constant renames applied (where needed)."
fi

# Vocabulary
if [[ ! -f "$ORB_DIR/Vocabulary/ORBvoc.txt" ]]; then
    if [[ -f "$ORB_DIR/Vocabulary/ORBvoc.txt.tar.gz" ]]; then
        log "Extracting ORB vocabulary (~145 MB, takes a moment)"
        tar -xzf "$ORB_DIR/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB_DIR/Vocabulary/"
        ok "Vocabulary extracted."
    else
        die "ORBvoc.txt.tar.gz missing from $ORB_DIR/Vocabulary/ — clone incomplete?"
    fi
else
    ok "Vocabulary already extracted."
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: ORB-SLAM3 build
# ─────────────────────────────────────────────────────────────────────────────
if [[ $ONLY_EXT -eq 0 ]]; then
log "Stage 4/5: Building ORB-SLAM3 (this is the long one — 20-40 min on 2 cores)"

if [[ -f "$ORB_DIR/lib/libORB_SLAM3.so" ]]; then
    ok "libORB_SLAM3.so already built — skipping."
else
    cd "$ORB_DIR"
    chmod +x build.sh
    # Reduce build.sh's parallelism to match JOBS — the stock script uses `make -j`
    # (unlimited), which OOMs on WSL2 with default memory limits.
    sed -i "s/make -j\$/make -j$JOBS/; s/make -j /make -j$JOBS /" build.sh || true
    ./build.sh 2>&1 | tee "$BUILD_DIR/orbslam3_build.log"

    [[ -f "$ORB_DIR/lib/libORB_SLAM3.so" ]] || die \
"ORB-SLAM3 build did NOT produce lib/libORB_SLAM3.so.
Full log: $BUILD_DIR/orbslam3_build.log
Last 40 lines:
$(tail -40 "$BUILD_DIR/orbslam3_build.log" 2>/dev/null)"
    ok "ORB-SLAM3 built."
fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: our mono_video runner
# ─────────────────────────────────────────────────────────────────────────────
log "Stage 5/5: Building mono_video"

[[ -f "$ORB_DIR/lib/libORB_SLAM3.so" ]] || die \
    "libORB_SLAM3.so not found at $ORB_DIR/lib/ — run without --only-ext first."

EXT_SRC="$PROJECT_ROOT/orbslam_ext"
EXT_BUILD="$BUILD_DIR/orbslam_ext_build"

[[ -f "$EXT_SRC/mono_video.cc" ]] || die "mono_video.cc not found at $EXT_SRC"

mkdir -p "$EXT_BUILD"
cd "$EXT_BUILD"
cmake "$EXT_SRC" -DORB_SLAM3_DIR="$ORB_DIR" -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tee cmake.log
make -j"$JOBS" 2>&1 | tee build.log

[[ -x "$EXT_BUILD/mono_video" ]] || die \
"mono_video did not build. Log: $EXT_BUILD/build.log
Last 40 lines:
$(tail -40 "$EXT_BUILD/build.log" 2>/dev/null)"

ok "mono_video built -> $EXT_BUILD/mono_video"

# ─────────────────────────────────────────────────────────────────────────────
log "BUILD COMPLETE"
echo "  ORB-SLAM3 lib : $ORB_DIR/lib/libORB_SLAM3.so"
echo "  Vocabulary    : $ORB_DIR/Vocabulary/ORBvoc.txt"
echo "  mono_video    : $EXT_BUILD/mono_video"
[[ -f "$NOTES" ]] && echo "  Patches applied are recorded in: $NOTES"
echo ""
echo "Next: python3 run_pipeline.py --video IMG_1112 --videos_dir <dir> --output_dir results/IMG_1112_run"
