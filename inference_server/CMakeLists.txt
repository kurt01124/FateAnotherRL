cmake_minimum_required(VERSION 3.18)
project(fate_inference_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA Toolkit (provides CUDA::nvToolsExt target) ---
find_package(CUDAToolkit QUIET)

# nvToolsExt was removed in CUDA 12+; provide a stub if missing
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

# --- LibTorch ---
find_package(Torch REQUIRED)

# --- Executable ---
add_executable(fate_inference_server
    src/main.cpp
    src/udp_server.cpp
    src/state_encoder.cpp
    src/inference_engine.cpp
    src/reward_calc.cpp
    src/rollout_writer.cpp
)

target_include_directories(fate_inference_server PRIVATE include)
target_link_libraries(fate_inference_server "${TORCH_LIBRARIES}")

# Ensure ABI compatibility
set_property(TARGET fate_inference_server PROPERTY CXX_STANDARD 17)

# --- Platform-specific ---
if(WIN32)
    target_link_libraries(fate_inference_server ws2_32)
endif()

# --- Copy LibTorch DLLs on Windows (MSVC) ---
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET fate_inference_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:fate_inference_server>
    )
endif()

# --- Compiler flags ---
if(MSVC)
    target_compile_options(fate_inference_server PRIVATE /W3 /O2)
    target_compile_definitions(fate_inference_server PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(fate_inference_server PRIVATE -Wall -Wextra -O2)
endif()
