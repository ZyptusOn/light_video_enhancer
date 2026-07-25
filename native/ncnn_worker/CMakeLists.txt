cmake_minimum_required(VERSION 3.21)
project(lve_ncnn_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(NCNN_SOURCE_DIR "" CACHE PATH "Path to a current Tencent/ncnn checkout")
if(NOT EXISTS "${NCNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set NCNN_SOURCE_DIR to a Tencent/ncnn checkout")
endif()

set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_SIMPLEVK ON CACHE BOOL "" FORCE)
set(NCNN_VULKAN_ONLINE_SPIRV ON CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION ON CACHE BOOL "" FORCE)

# NCNN enables almost every neural-network layer by default.  The three
# bundled model families use a small union of them; building only that union
# cuts the worker/package size without changing model execution.
file(STRINGS "${NCNN_SOURCE_DIR}/src/CMakeLists.txt" NCNN_LAYER_LINES
    REGEX "^ncnn_add_layer\\(")
foreach(LINE IN LISTS NCNN_LAYER_LINES)
    string(REGEX REPLACE "^ncnn_add_layer\\(([A-Za-z0-9_]+).*$" "\\1"
        LAYER_CLASS "${LINE}")
    string(TOLOWER "${LAYER_CLASS}" LAYER_NAME)
    set(WITH_LAYER_${LAYER_NAME} OFF CACHE BOOL "" FORCE)
endforeach()
set(LVE_REQUIRED_NCNN_LAYERS
    concat convolution crop deconvolution eltwise flatten innerproduct input
    pooling prelu reduction relu scale sigmoid slice split binaryop unaryop
    padding interp packing cast pixelshuffle clip gemm)
foreach(LAYER_NAME IN LISTS LVE_REQUIRED_NCNN_LAYERS)
    set(WITH_LAYER_${LAYER_NAME} ON CACHE BOOL "" FORCE)
endforeach()

add_subdirectory("${NCNN_SOURCE_DIR}" ncnn)

function(embed_shader SHADER)
    get_filename_component(SHADER_NAME "${SHADER}" NAME_WE)
    set(HEADER "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.comp.hex.h")
    add_custom_command(
        OUTPUT "${HEADER}"
        COMMAND "${CMAKE_COMMAND}"
            "-DSHADER_SRC=${SHADER}"
            "-DSHADER_COMP_HEADER=${HEADER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_comp_header.cmake"
        DEPENDS "${SHADER}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_shader_comp_header.cmake"
        VERBATIM)
    set_property(GLOBAL APPEND PROPERTY LVE_SHADER_HEADERS "${HEADER}")
endfunction()

file(GLOB RIFE_SHADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/rife/*.comp")
file(GLOB CUGAN_SHADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/realcugan/*.comp")
file(GLOB ESRGAN_SHADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/realesrgan/*.comp")
foreach(SHADER IN LISTS RIFE_SHADERS CUGAN_SHADERS ESRGAN_SHADERS)
    embed_shader("${SHADER}")
endforeach()
get_property(LVE_SHADER_HEADERS GLOBAL PROPERTY LVE_SHADER_HEADERS)
add_custom_target(lve_ncnn_shaders DEPENDS ${LVE_SHADER_HEADERS})

add_executable(lve-ncnn-worker
    worker.cpp
    upstream/rife/rife.cpp
    upstream/rife/warp.cpp
    upstream/realcugan/realcugan.cpp
    upstream/realesrgan/realesrgan.cpp
    ${LVE_SHADER_HEADERS})
add_dependencies(lve-ncnn-worker lve_ncnn_shaders)

target_include_directories(lve-ncnn-worker PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/rife"
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/realcugan"
    "${CMAKE_CURRENT_SOURCE_DIR}/upstream/realesrgan")
target_link_libraries(lve-ncnn-worker PRIVATE ncnn)
target_compile_definitions(lve-ncnn-worker PRIVATE
    WINVER=0x0601 _WIN32_WINNT=0x0601 NOMINMAX WIN32_LEAN_AND_MEAN)

if(MSVC)
    target_compile_options(lve-ncnn-worker PRIVATE /O2 /GL /utf-8 /EHsc)
    target_link_options(lve-ncnn-worker PRIVATE /LTCG)
endif()

install(TARGETS lve-ncnn-worker RUNTIME DESTINATION .)
