CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ICGRF-celldetection-class)

SET( RUN_TEST FALSE )
SET ( DEBUG_MODE ON )
SET ( CMAKE_ECLIPSE_MAKE_ARGUMENTS "${CMAKE_ECLIPSE_MAKE_ARGUMENTS} -j8" )
SET ( CMAKE_CXX_FLAGS "-g" )

# open cv ===================================================================
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
  MESSAGE(STATUS "OpenCV ${OpenCV_VERSION} was found.")
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
  # custom output
  MESSAGE(STATUS "Found libraries: ${OpenCV_LIBRARIES}")
ELSE (OpenCV_FOUND)
  MESSAGE(FATAL_ERROR "OpenCV was not found.")
ENDIF (OpenCV_FOUND)

# Boost ===================================================================
FIND_PACKAGE(Boost COMPONENTS serialization system chrono regex timer filesystem iostreams program_options REQUIRED)
IF (Boost_FOUND)
  MESSAGE(STATUS "Boost ${Boost_VERSION} was found.")
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ELSE (Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost was not found.")
ENDIF (Boost_FOUND)

# OpenMP
FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
  MESSAGE(STATUS "OpenMP was found.")
ELSE (OPENMP_FOUND)
  MESSAGE(FATAL_ERROR "OpenMP was not found.")
ENDIF (OPENMP_FOUND)


# Custom/Other Stuff ======================================================
SET(CUSTOM_LOCAL_INCLUDE /usr/local/include)
SET(CUSTOM_LOCAL_LIB_DIRS /usr/local/lib)


ADD_DEFINITIONS(-O3 -mtune=native -msse3 -mfpmath=sse -fopenmp ${OpenMP_CXX_FLAGS})

INCLUDE_DIRECTORIES (
    ${CUSTOM_LOCAL_INCLUDE}
    ${OPENNI_INCLUDES}
)

LINK_DIRECTORIES (
    ${CUSTOM_LOCAL_LIB_DIRS}
    ${Boost_LIBRARY_DIRS}
    ${OpenCV_LINK_DIRECTORIES}
)


# SOURCE FILES
FILE ( GLOB APP_HEADERS *.h )
FILE ( GLOB APP_SOURCES *.cpp )
FILE ( GLOB LIB_HEADERS ../../rflib/*.h )
FILE ( GLOB LIB_SOURCES ../../rflib/*.cpp )
FILE ( GLOB LIBCORE_HEADERS ../../rflib/core/*.h )
FILE ( GLOB LIBCORE_SOURCES ../../rflib/core/*.cpp )

# INCLUDE THE CORE HEADERS
INCLUDE_DIRECTORIES ( ../../rflib )
INCLUDE_DIRECTORIES ( ../../rflib/core )

# CORE LIBRARY DECLARATION
ADD_LIBRARY ( icg_rf_core  
	${LIB_HEADERS}
	${LIB_SOURCES}
	${LIBCORE_HEADERS}
	${LIBCORE_SOURCES}
)

# ADD THE EXECUTABLE
ADD_EXECUTABLE ( icgrf-celldet-class 
	${APP_HEADERS}
	${APP_SOURCES}
)

TARGET_LINK_LIBRARIES(icgrf-celldet-class
        icg_rf_core
        config++
        gomp
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)



