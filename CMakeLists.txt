cmake_minimum_required (VERSION 3.1)

project (DGM)

# The version number
set (DGM_VERSION_MAJOR 1)
set (DGM_VERSION_MINOR 5)
set (DGM_VERSION_PATCH 2)
set (DGM_VERSION ${DGM_VERSION_MAJOR}.${DGM_VERSION_MINOR}.${DGM_VERSION_PATCH})

configure_file(${PROJECT_SOURCE_DIR}/cmake/types.h.in ${PROJECT_SOURCE_DIR}/include/types.h)

# OpenCV
find_package(OpenCV 3 REQUIRED core features2d highgui imgproc imgcodecs ml PATHS "$ENV{OPENCVDIR}/build")
# find_package(OpenCV 3 REQUIRED PATHS "$ENV{OPENCVDIR}/build")

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Sets
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(DGM_LIB $<$<CONFIG:Debug>:dgm${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d.lib>$<$<CONFIG:Release>:dgm${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}.lib>)
set(FEX_LIB $<$<CONFIG:Debug>:fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d.lib>$<$<CONFIG:Release>:fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}.lib>)
set(VIS_LIB $<$<CONFIG:Debug>:vis${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d.lib>$<$<CONFIG:Release>:vis${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}.lib>)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")	
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")					
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install   CACHE PATH "Installation directory" FORCE)

#install
install(DIRECTORY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/etc DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${PROJECT_SOURCE_DIR}/License.txt DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/batch/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.bat")

#definitions
add_definitions(-D_CRT_SECURE_NO_WARNINGS 
				-D_SCL_SECURE_NO_WARNINGS
				)

# Options
option(DEBUG_PRINT_INFO "Output debug information" OFF)
option(DEBUG_MODE "Debugging mode" OFF)
option(ENABLE_PPL "Use Parallel Pattern Library for parallel CPU computing" ON) 
option(ENABLE_AMP "Use AMP Algorithms Library for parallel GPU computing" OFF) 
option(USE_OPENGL "Use OpenGL library" OFF) 
option(USE_SHERWOOD "Use Microsoft Sherwood Library for CTrainNodeMsRF class" ON)

if (USE_OPENGL)  
	#OpenGL  
	find_package(OpenGL REQUIRED)  
  
	# GLFW  
	set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)  
	set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)  
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)  
	set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)  
	
	add_subdirectory(3rdparty/glfw-3.2.1)  

	set_target_properties(glfw PROPERTIES FOLDER "3rdparty")
endif()  
 
 # Sub-directories where more CMakeLists.txt exist
add_subdirectory(modules/DGM)
add_subdirectory(modules/FEX)
add_subdirectory(modules/VIS)
add_subdirectory(demos)

# ===============================

# Add all targets to the build-tree export set
if(USE_OPENGL)
	export(TARGETS DGM FEX VIS glew glfw 
		FILE ${PROJECT_BINARY_DIR}/DGMTargets.cmake)
else()
	export(TARGETS DGM FEX VIS 
		FILE ${PROJECT_BINARY_DIR}/DGMTargets.cmake)
endif()
# Export the package for use from the build-tree
export(PACKAGE DGM)

# Create the FooBarConfig.cmake and FooBarConfigVersion files
# REL_INCLUDE_DIR = include
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(${PROJECT_SOURCE_DIR}/cmake/DGMConfig.cmake.in ${PROJECT_BINARY_DIR}/DGMConfig.cmake @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${DGM_CMAKE_DIR}/include")
configure_file(${PROJECT_SOURCE_DIR}/cmake/DGMConfig.cmake.in ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DGMConfig.cmake @ONLY)
# ... for both
configure_file(${PROJECT_SOURCE_DIR}/cmake/DGMConfigVersion.cmake.in ${PROJECT_BINARY_DIR}/DGMConfigVersion.cmake @ONLY)
 
# Install the FooBarConfig.cmake and FooBarConfigVersion.cmake
install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DGMConfig.cmake ${PROJECT_BINARY_DIR}/DGMConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
 
# Install the export set for use with the install-tree
install(EXPORT DGMTargets DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

