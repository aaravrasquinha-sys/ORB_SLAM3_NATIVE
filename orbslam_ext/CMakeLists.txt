cmake_minimum_required(VERSION 3.10)
project(orbslam_ext)

# Standalone build for mono_video.cc, linking against an ALREADY-BUILT
# ORB-SLAM3 tree. Keeping this out of the ORB_SLAM3 source tree means a
# failed/rebuilt ORB-SLAM3 doesn't clobber our runner, and our runner can be
# recompiled in seconds without re-running ORB-SLAM3's own build.sh.
#
# Required: -DORB_SLAM3_DIR=/path/to/ORB_SLAM3

if(NOT DEFINED ORB_SLAM3_DIR)
    message(FATAL_ERROR "Set -DORB_SLAM3_DIR=/path/to/ORB_SLAM3")
endif()

if(NOT EXISTS "${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so")
    message(FATAL_ERROR
        "libORB_SLAM3.so not found at ${ORB_SLAM3_DIR}/lib/. "
        "Build ORB-SLAM3 first (./build.sh in that directory).")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)          # ORB-SLAM3 master requires C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.0 REQUIRED)
endif()
message(STATUS "Using OpenCV ${OpenCV_VERSION}")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(mono_video mono_video.cc)

target_link_libraries(mono_video
    ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    -lboost_serialization
    -lcrypto
    -lpthread
)

# Embed the ORB-SLAM3 lib dirs so the binary runs without LD_LIBRARY_PATH
set_target_properties(mono_video PROPERTIES
    BUILD_RPATH "${ORB_SLAM3_DIR}/lib;${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib;${ORB_SLAM3_DIR}/Thirdparty/g2o/lib"
    INSTALL_RPATH "${ORB_SLAM3_DIR}/lib;${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib;${ORB_SLAM3_DIR}/Thirdparty/g2o/lib"
)
