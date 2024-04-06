cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dla)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)

file(GLOB SOURCES src/*.cpp)
add_executable(dla ${SOURCES})
target_link_libraries(dla "${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dla
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:dla>)
endif (MSVC)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/0af976647f49ff0944c5971ae0a45d6fcdf1ecca.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
file(GLOB TESTS test/test_*.cpp)
add_executable(tests ${TESTS} ${SOURCES})
target_link_libraries(tests gtest_main "${TORCH_LIBRARIES}")

include(GoogleTest)
gtest_discover_tests(tests)

enable_testing()