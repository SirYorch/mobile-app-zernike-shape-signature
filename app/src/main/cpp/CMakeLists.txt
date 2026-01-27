cmake_minimum_required(VERSION 3.22.1)
project(moments)

# ===== OpenCV Android SDK =====
set(OpenCV_DIR
    ${CMAKE_SOURCE_DIR}/opencv/sdk/native/jni
)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# ===== Librería nativa =====
add_library(
    moments
    SHARED
    native-lib.cpp
)

find_library(
    log-lib
    log
)

target_link_libraries(
    moments
    ${OpenCV_LIBS}
    ${log-lib}
    jnigraphics
)

