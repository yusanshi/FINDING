cmake_minimum_required(VERSION 3.8)
project(client)

set(CMAKE_CXX_STANDARD 11)

include_directories(../lib_fhesi)
include_directories(/usr/include/NTL)
set(SOURCE_FILES main.cpp clientfhesiutils.cpp clientfhesiutils.h KClientT2V1.h KClientT2V1.cpp)

add_executable(client ${SOURCE_FILES})
target_link_libraries(client
 ${CMAKE_CURRENT_SOURCE_DIR}/../lib_fhesi/build/libfhesi.a
 ${CMAKE_CURRENT_SOURCE_DIR}/../lib_ntl/src/ntl.a 
 libgmp.a libm.a)

