cmake_minimum_required (VERSION 3.1)

project (DGM)

# The version number
set (DGM_VERSION_MAJOR 1)
set (DGM_VERSION_MINOR 5)
set (DGM_VERSION_PATCH 0)

configure_file(${PROJECT_SOURCE_DIR}/cmake/types.h.in ${PROJECT_SOURCE_DIR}/include/types.h)

# OpenCV
find_package(OpenCV 2 REQUIRED core highgui imgproc ml PATHS "$ENV{OPENCVDIR}/build")

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Sets
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(DGM_LIB $<$<CONFIG:Debug>:dgm${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d.lib>$<$<CONFIG:Release>:dgm${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}.lib>)
set(FEX_LIB $<$<CONFIG:Debug>:fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d.lib>$<$<CONFIG:Release>:fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}.lib>)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")	
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")					
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 
set(PROJECT_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install)

#install
install(DIRECTORY ${PROJECT_SOURCE_DIR}/data DESTINATION ${PROJECT_INSTALL_PREFIX})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/etc DESTINATION ${PROJECT_INSTALL_PREFIX})
install(FILES ${PROJECT_SOURCE_DIR}/License.txt DESTINATION ${PROJECT_INSTALL_PREFIX})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/batch/ DESTINATION ${PROJECT_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.bat")

#definitions
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)

# Options
option(DEBUG_PRINT_INFO "Output debug information" OFF)
option(DEBUG_MODE "Debugging mode" OFF)
option(ENABLE_PPL "Use Parallel Pattern Library for parallel CPU computing" ON) 
option(ENABLE_AMP "Use AMP Algorithms Library for parallel GPU computing" OFF) 
option(USE_SHERWOOD "Use Microsoft Sherwood Library for CTrainNodeMsRF class" ON)

# Sub-directories where more CMakeLists.txt exist
add_subdirectory(modules/DGM)
add_subdirectory(modules/FEX)
add_subdirectory(demos)
