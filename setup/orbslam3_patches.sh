#!/usr/bin/env bash
# orbslam3_patches.sh — known source patches needed to compile UZ-SLAMlab/ORB_SLAM3
# on modern Ubuntu/GCC (tested on Ubuntu 22.04, GCC 11/12). Sourced by
# build_orbslam3.sh. Each patch function is idempotent (checks whether the fix is
# already present before touching a file). Every patch is recorded in BUILD_NOTES.md.
#
# To add a new patch discovered during a build failure:
#   1. Write a patch_* function below
#   2. Call it from apply_all_patches()
#   3. Run build_orbslam3.sh again; it will apply only the new patch

# ── Helper: add an #include if not already present ───────────────────────────
_add_include() {
  local file="$1"
  local header="$2"         # e.g.  "thread"
  local include_line="#include <${header}>"
  if grep -qF "$include_line" "$file"; then
    return 0   # already present
  fi
  # Insert after the first existing #include line
  sed -i "0,/^#include/s|^#include|${include_line}\n#include|" "$file"
  echo "    [patch] Added $include_line to $file"
}

# ── Helper: replace a string in a file if old string still present ───────────
_sed_replace() {
  local file="$1"
  local old_str="$2"
  local new_str="$3"
  if grep -qF "$old_str" "$file"; then
    sed -i "s|${old_str}|${new_str}|g" "$file"
    echo "    [patch] Replaced '${old_str}' -> '${new_str}' in $file"
  fi
}

# ── Patch: add missing #include <thread> ─────────────────────────────────────
# Error:   'std::thread' was not declared in this scope
# Affected: LocalMapping.cc, LoopClosing.cc, System.cc, Tracking.cc
patch_missing_thread_include() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local patched=0
  for f in \
    "$ORB_DIR/src/LocalMapping.cc" \
    "$ORB_DIR/src/LoopClosing.cc" \
    "$ORB_DIR/src/System.cc" \
    "$ORB_DIR/src/Tracking.cc"
  do
    if [[ -f "$f" ]]; then
      if ! grep -q '#include <thread>' "$f"; then
        _add_include "$f" "thread"
        patched=1
      fi
    fi
  done
  if [[ $patched -eq 1 ]]; then
    cat >> "$NOTES_FILE" <<'EOF'

### Missing #include <thread>
**Files:** src/LocalMapping.cc, src/LoopClosing.cc, src/System.cc, src/Tracking.cc

**Error:** `'std::thread' was not declared in this scope` (GCC 11+ is stricter
about transitive includes).

**Fix:** Added `#include <thread>` to each affected file before the first existing
`#include` directive.
EOF
  fi
}

# ── Patch: add missing #include <unistd.h> ───────────────────────────────────
# Error:   'usleep' was not declared in this scope
# Affected: LocalMapping.cc, LoopClosing.cc
patch_missing_unistd_include() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local patched=0
  for f in \
    "$ORB_DIR/src/LocalMapping.cc" \
    "$ORB_DIR/src/LoopClosing.cc" \
    "$ORB_DIR/src/Viewer.cc"
  do
    if [[ -f "$f" ]]; then
      if ! grep -q '#include <unistd.h>' "$f"; then
        _add_include "$f" "unistd.h"
        patched=1
      fi
    fi
  done
  if [[ $patched -eq 1 ]]; then
    cat >> "$NOTES_FILE" <<'EOF'

### Missing #include <unistd.h>
**Files:** src/LocalMapping.cc, src/LoopClosing.cc, src/Viewer.cc

**Error:** `'usleep' was not declared in this scope`. GCC 11 no longer pulls
<unistd.h> in transitively from <pthread.h>.

**Fix:** Added `#include <unistd.h>` to each affected file.
EOF
  fi
}

# ── Patch: add missing #include <chrono> in examples ─────────────────────────
# Error:   'std::chrono' has not been declared
# Affected: Examples/Monocular/mono_tum.cc and similar
patch_missing_chrono_include() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local patched=0
  for f in \
    "$ORB_DIR/Examples/Monocular/mono_tum.cc" \
    "$ORB_DIR/Examples/Monocular/mono_euroc.cc" \
    "$ORB_DIR/Examples/Monocular/mono_kitti.cc"
  do
    if [[ -f "$f" ]]; then
      if ! grep -q '#include <chrono>' "$f"; then
        _add_include "$f" "chrono"
        patched=1
      fi
    fi
  done
  if [[ $patched -eq 1 ]]; then
    cat >> "$NOTES_FILE" <<'EOF'

### Missing #include <chrono> in Examples
**Files:** Examples/Monocular/*.cc

**Error:** `'std::chrono' has not been declared` in example binaries.

**Fix:** Added `#include <chrono>` to affected example files.
EOF
  fi
}

# ── Patch: add public GetAtlas() accessor to System.h ────────────────────────
# Needed so mono_video.cc can retrieve all map points after SLAM.Shutdown().
# Error:   'mpAtlas' is private/protected within System class
# Fix:     Add a public getter method.
patch_system_get_atlas() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local SYS_H="$ORB_DIR/include/System.h"
  if [[ ! -f "$SYS_H" ]]; then
    return 0
  fi
  # Check if GetAtlas() is already present (either added by us or by the fork)
  if grep -q 'GetAtlas()' "$SYS_H"; then
    return 0
  fi
  # Insert after the MapChanged() declaration (or after the last public method
  # near it). We look for "bool MapChanged();" and add GetAtlas right after.
  if grep -q 'bool MapChanged' "$SYS_H"; then
    sed -i '/bool MapChanged/a\    // Added by orbslam3_patches.sh for map-point access\n    Atlas* GetAtlas(){return mpAtlas;}' "$SYS_H"
  else
    # Fallback: insert before the closing brace of the public block by finding
    # the line with "void SaveAtlas" or similar — last resort sed approach.
    sed -i '/void SaveAtlas/a\    Atlas* GetAtlas(){return mpAtlas;}' "$SYS_H"
  fi
  echo "    [patch] Added GetAtlas() public accessor to include/System.h"
  cat >> "$NOTES_FILE" <<'EOF'

### System::GetAtlas() public accessor
**File:** include/System.h

**Error:** mono_video.cc needs all map points for map/merged.ply but mpAtlas is
private. The standard public API (SaveMap, SaveTrajectoryTUM) doesn't expose raw
MapPoint* pointers needed for custom PLY output.

**Fix:** Added `Atlas* GetAtlas(){return mpAtlas;}` as a public inline method in
System.h, immediately after the `bool MapChanged()` declaration. This is a
minimal, non-breaking addition that the community commonly applies.
EOF
}

# ── Patch: OpenCV 4 — CV_LOAD_IMAGE_UNCHANGED → cv::IMREAD_UNCHANGED ─────────
# Error:   'CV_LOAD_IMAGE_UNCHANGED' was not declared (removed in OpenCV 4)
patch_opencv4_imread_flags() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local patched=0
  # Search recursively for the old constant name
  while IFS= read -r f; do
    if grep -q 'CV_LOAD_IMAGE_UNCHANGED' "$f"; then
      sed -i 's/CV_LOAD_IMAGE_UNCHANGED/cv::IMREAD_UNCHANGED/g' "$f"
      echo "    [patch] CV_LOAD_IMAGE_UNCHANGED -> cv::IMREAD_UNCHANGED in $f"
      patched=1
    fi
    if grep -q 'CV_LOAD_IMAGE_GRAYSCALE' "$f"; then
      sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::IMREAD_GRAYSCALE/g' "$f"
      echo "    [patch] CV_LOAD_IMAGE_GRAYSCALE -> cv::IMREAD_GRAYSCALE in $f"
      patched=1
    fi
  done < <(grep -rl 'CV_LOAD_IMAGE' "$ORB_DIR/src" "$ORB_DIR/Examples" 2>/dev/null)
  if [[ $patched -eq 1 ]]; then
    cat >> "$NOTES_FILE" <<'EOF'

### OpenCV 4: deprecated CV_LOAD_IMAGE_* constants
**Error:** `'CV_LOAD_IMAGE_UNCHANGED' was not declared in this scope`. OpenCV 4
removed these C-style constants.

**Fix:** Replaced with the scoped enum equivalents: `cv::IMREAD_UNCHANGED`,
`cv::IMREAD_GRAYSCALE`, etc.
EOF
  fi
}

# ── Patch: Sophus deprecated() / [[nodiscard]] compile warnings-as-errors ─────
# If the build uses -Werror and Sophus emits deprecated warnings, they can
# break the build. Fix: remove -Werror from the ORB-SLAM3 CMakeLists if present.
patch_remove_werror() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local CMAKE="$ORB_DIR/CMakeLists.txt"
  if [[ ! -f "$CMAKE" ]]; then
    return 0
  fi
  if grep -q '\-Werror' "$CMAKE"; then
    sed -i 's/-Werror//g' "$CMAKE"
    echo "    [patch] Removed -Werror from $CMAKE"
    cat >> "$NOTES_FILE" <<'EOF'

### Removed -Werror from CMakeLists.txt
**Error:** Sophus and/or OpenCV 4 headers emit deprecation warnings that fail the
build when -Werror is active.

**Fix:** Removed -Werror from ORB_SLAM3/CMakeLists.txt. Warnings remain visible
but do not abort compilation.
EOF
  fi
}

# ── Patch: Thirdparty g2o — missing #include <unistd.h> ──────────────────────
# Error:   'usleep'/'sleep' not declared in g2o's solvers on GCC 11+
patch_g2o_unistd() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local patched=0
  local G2O_DIR="$ORB_DIR/Thirdparty/g2o"
  if [[ ! -d "$G2O_DIR" ]]; then
    return 0
  fi
  while IFS= read -r f; do
    if ! grep -q '#include <unistd.h>' "$f"; then
      _add_include "$f" "unistd.h"
      patched=1
    fi
  done < <(grep -rl 'usleep\|::sleep(' "$G2O_DIR" --include='*.cc' --include='*.cpp' 2>/dev/null)
  if [[ $patched -eq 1 ]]; then
    cat >> "$NOTES_FILE" <<'EOF'

### Thirdparty/g2o missing #include <unistd.h>
**Error:** `'usleep' was not declared` in g2o optimizer source files.

**Fix:** Added `#include <unistd.h>` to each affected g2o .cc/.cpp file.
EOF
  fi
}

# ── Patch: C++14 standard flag in Thirdparty CMakeLists ──────────────────────
# Some Thirdparty components default to C++11 which breaks with GCC 11+ and
# Sophus (which requires C++14). Ensure CMAKE_CXX_STANDARD = 14.
patch_cxx14_thirdparty() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local patched=0
  for cmake_file in \
    "$ORB_DIR/Thirdparty/DBoW2/CMakeLists.txt" \
    "$ORB_DIR/Thirdparty/g2o/CMakeLists.txt"
  do
    if [[ -f "$cmake_file" ]]; then
      if grep -q 'CMAKE_CXX_STANDARD 11' "$cmake_file"; then
        sed -i 's/CMAKE_CXX_STANDARD 11/CMAKE_CXX_STANDARD 14/g' "$cmake_file"
        echo "    [patch] C++11 -> C++14 in $cmake_file"
        patched=1
      elif ! grep -q 'CMAKE_CXX_STANDARD' "$cmake_file"; then
        # No standard set at all — prepend it
        sed -i '1s|^|set(CMAKE_CXX_STANDARD 14)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n|' "$cmake_file"
        echo "    [patch] Added CMAKE_CXX_STANDARD 14 to $cmake_file"
        patched=1
      fi
    fi
  done
  if [[ $patched -eq 1 ]]; then
    cat >> "$NOTES_FILE" <<'EOF'

### C++ standard upgrade: 11 -> 14 in Thirdparty
**Error:** Compilation errors in DBoW2 or g2o with GCC 11+ when C++11 is set.
Sophus (a dependency) requires C++14 features.

**Fix:** Changed CMAKE_CXX_STANDARD from 11 to 14 in Thirdparty CMakeLists.txt.
EOF
  fi
}

# ── Patch: OpenMP flag mismatch with Eigen/Sophus ────────────────────────────
# Some Eigen operations with GCC 11+ need to explicitly link against libpthread.
# This is usually handled by CMake's find_package(Threads) but may be missing.
patch_eigen_pthread() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"
  local CMAKE="$ORB_DIR/CMakeLists.txt"
  if [[ ! -f "$CMAKE" ]]; then
    return 0
  fi
  if ! grep -q 'find_package(Threads' "$CMAKE" && ! grep -q 'Threads::Threads' "$CMAKE"; then
    # Add after the first project() declaration
    sed -i '/^project(/a find_package(Threads REQUIRED)' "$CMAKE"
    echo "    [patch] Added find_package(Threads REQUIRED) to root CMakeLists.txt"
    cat >> "$NOTES_FILE" <<'EOF'

### Missing find_package(Threads) in root CMakeLists.txt
**Error:** Linker errors related to pthread on some Ubuntu configurations where
Threads is not found transitively.

**Fix:** Added `find_package(Threads REQUIRED)` after the project() declaration
in the root CMakeLists.txt.
EOF
  fi
}

# ── Master entry point called by build_orbslam3.sh ───────────────────────────
apply_all_patches() {
  local ORB_DIR="$1"
  local NOTES_FILE="$2"

  echo "  Applying patches (each is idempotent — safe to re-run):"
  patch_remove_werror        "$ORB_DIR" "$NOTES_FILE"
  patch_cxx14_thirdparty     "$ORB_DIR" "$NOTES_FILE"
  patch_missing_thread_include "$ORB_DIR" "$NOTES_FILE"
  patch_missing_unistd_include "$ORB_DIR" "$NOTES_FILE"
  patch_missing_chrono_include "$ORB_DIR" "$NOTES_FILE"
  patch_opencv4_imread_flags "$ORB_DIR" "$NOTES_FILE"
  patch_g2o_unistd           "$ORB_DIR" "$NOTES_FILE"
  patch_system_get_atlas     "$ORB_DIR" "$NOTES_FILE"
  patch_eigen_pthread        "$ORB_DIR" "$NOTES_FILE"
  echo "  All patches applied (already-applied patches were skipped)."
}
