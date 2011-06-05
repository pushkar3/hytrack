# The name of our project is "hytrack"

cmake_minimum_required (VERSION 2.6)
project (hytrack)

SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "1")
SET(CPACK_PACKAGE_NAME "hytrack")
SET(CPACK_SYSTEM_NAME "i386")

set (CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wl,--no-undefined -fopenmp")

find_package ( OpenCV REQUIRED )

set ( SRC
main
)
 
include_directories ( ${OpenCV_INCLUDE_DIRS} )
add_executable (hytrack ${SRC})
target_link_libraries (hytrack m)

