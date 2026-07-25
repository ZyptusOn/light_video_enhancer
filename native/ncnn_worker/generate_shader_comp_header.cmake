# Embed GLSL text. NCNN compiles it once at worker startup and can reuse the
# resulting Vulkan pipelines for every video batch.
file(READ "${SHADER_SRC}" comp_data)
string(FIND "${comp_data}" "#version" version_start)
if(version_start LESS 0)
    message(FATAL_ERROR "Shader has no #version directive: ${SHADER_SRC}")
endif()
string(SUBSTRING "${comp_data}" ${version_start} -1 comp_data)
string(REGEX REPLACE "\n +" "\n" comp_data "${comp_data}")
get_filename_component(shader_name "${SHADER_SRC}" NAME_WE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.text2hex.txt" "${comp_data}")
file(READ "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.text2hex.txt" comp_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," comp_hex "${comp_hex}")
string(FIND "${comp_hex}" "," tail REVERSE)
string(SUBSTRING "${comp_hex}" 0 ${tail} comp_hex)
file(WRITE "${SHADER_COMP_HEADER}"
    "static const char ${shader_name}_comp_data[] = {${comp_hex}};\n")
