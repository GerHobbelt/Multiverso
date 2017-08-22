set(MULTIVERSO_DIR ${PROJECT_SOURCE_DIR})
set(KMEANS ${PROJECT_SOURCE_DIR}/Applications/KMeans)

find_package(MPI REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

if(LOGLEVEL STREQUAL "DEBUG")
  add_definitions(-DLOGLEVEL_DEBUG)
elseif(LOGLEVEL STREQUAL "FATAL")
  add_definitions(-DLOGLEVEL_FATAL)
elseif(LOGLEVEL STREQUAL "ERROR")
  add_definitions(-DLOGLEVEL_ERROR)
else()
  add_definitions(-DLOGLEVEL_INFO)
endif()

set(MULTIVERSO_INC ${MULTIVERSO_DIR}/include)
set(MULTIVERSO_LIB ${MULTIVERSO_DIR}/build/src)
set(MULTIVERSO_SRC ${MULTIVERSO_DIR}/src)

include_directories(${MULTIVERSO_INC})
include_directories(${KMEANS}/src)

link_directories(${MULTIVERSO_LIB})

set(SRCDIR ${KMEANS}/src)
aux_source_directory(${KMEANS}/src SRC_ROOT)
aux_source_directory(${SRCDIR}/util SRC_UTIL)
set(SRC ${SRC_UTIL} ${MULTIVERSO_SRC}/table/array_table.cpp ${SRC_ROOT})

add_executable(KMeans ${SRC})

target_link_libraries(KMeans multiverso ${MPI_CXX_LIBRARIES} pthread hdfs3 jvm)
