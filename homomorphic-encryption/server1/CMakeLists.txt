cmake_minimum_required(VERSION 3.8)
project(server1)

set(CMAKE_CXX_STANDARD 11)

include_directories(../lib_fhesi)
include_directories(/usr/include/NTL)
set(SOURCE_FILES main.cpp userverfhesiutils.cpp userverfhesiutils.h UServerT2V1.cpp UServerT2V1.h)

add_executable(server1 ${SOURCE_FILES})
target_link_libraries(server1 
 ${CMAKE_CURRENT_SOURCE_DIR}/../lib_fhesi/build/libfhesi.a
 ${CMAKE_CURRENT_SOURCE_DIR}/../lib_ntl/src/ntl.a 
 libgmp.a libm.a)
 