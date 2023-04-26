cmake_minimum_required(VERSION 3.0)
project(rwkv-server)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("Microsoft.ML.OnnxRuntime.DirectML.1.14.1/build/native/include")
link_directories("Microsoft.ML.OnnxRuntime.DirectML.1.14.1/runtimes/win-x64/native")

include_directories("tokenizer/bindings/c/include")
link_directories("tokenizer/release")

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/model)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB MODLE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp)

set(DEP_LIBS "${TORCH_LIBRARIES}" onnxruntime "../tokenizer/release/tokenizers.dll")

add_executable(rwkv-server main.cpp ${SRCS} ${MODLE_SRCS} )


target_link_libraries(rwkv-server ${DEP_LIBS})
set_property(TARGET rwkv-server PROPERTY CXX_STANDARD 17)

set_target_properties(rwkv-server PROPERTIES
                      INSTALL_RPATH "$ORIGIN/lib")
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

    add_custom_command(TARGET rwkv-server
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:rwkv-server>)
endif(MSVC)