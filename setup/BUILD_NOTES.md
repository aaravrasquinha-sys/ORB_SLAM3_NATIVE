# ORB-SLAM3 Build Notes

Every patch applied to get ORB-SLAM3 / Pangolin compiling on this machine's
Ubuntu + GCC version, recorded as it happened, so the build is reproducible.
Generated/appended to by `setup/build_orbslam3.sh`.
OS: Ubuntu 24.04.4 LTS
GCC: gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
CMake: cmake version 3.28.3


### Pangolin: missing #include <cstdint> in image_io_jpg.cpp
**Error:** 'uint8_t' was not declared in this scope (GCC 13+ no longer transitively includes <cstdint> via <jpeglib.h>).

**Fix:** Added '#include <cstdint>' to src/image/image_io_jpg.cpp before the first existing #include.
