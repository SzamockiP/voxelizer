include(FetchContent)

# GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.8
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# CLI11
add_library(CLI11 INTERFACE)
target_include_directories(CLI11 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/CLI11)
add_library(CLI11::CLI11 ALIAS CLI11)

# tinyobjloader
add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader)
add_library(tinyobjloader::tinyobjloader ALIAS tinyobjloader)