cmake_minimum_required(VERSION 3.25)
project(voxelizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED COMPONENTS glslc)

include(FetchContent)
FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc13
    GIT_SHALLOW TRUE
    SYSTEM)
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
    SYSTEM) # third-party headers stay out of our warnings
FetchContent_MakeAvailable(tinyobjloader vma)

# Shaders: GLSL -> SPIR-V as a C initializer list, #included into vk.cpp.
set(SHADERS tri.vert bin.frag voxel.frag compact_scan_groups.comp compact_scan_totals.comp compact_write.comp)
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
foreach(shader ${SHADERS})
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${shader}.inc)
    add_custom_command(
        OUTPUT ${out}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -mfmt=num -o ${out} ${SHADER_DIR}/${shader}
        DEPENDS ${SHADER_DIR}/${shader} ${SHADER_DIR}/draw.glsl ${SHADER_DIR}/compact.glsl
        VERBATIM)
    list(APPEND SPIRV ${out})
endforeach()

add_executable(voxelizer src/main.cpp src/vk.cpp src/voxelizer.cpp ${SPIRV})
target_include_directories(voxelizer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(voxelizer PRIVATE Vulkan::Vulkan tinyobjloader GPUOpen::VulkanMemoryAllocator)
if(MSVC)
    target_compile_options(voxelizer PRIVATE /utf-8 /W4)
    target_compile_definitions(voxelizer PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(voxelizer PRIVATE -Wall -Wextra)
endif()

enable_testing()
add_executable(test_voxelizer tests/test_voxelizer.cpp)
add_test(NAME voxelizer COMMAND test_voxelizer $<TARGET_FILE:voxelizer> ${CMAKE_CURRENT_BINARY_DIR}/testdata)
