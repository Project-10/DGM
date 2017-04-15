file(GLOB VIS_INCLUDE	${PROJECT_SOURCE_DIR}/include/VIS.h)
file(GLOB VIS_SOURCES	"*.cpp")
file(GLOB VIS_HEADERS	"*.h")
if(USE_OPENGL)
file(GLOB VIS_SHADERS	"*.glsl")
file(GLOB VIS_SOURCES	${VIS_SOURCES} "./Trackball Camera/*.cpp")
file(GLOB VIS_HEADERS	${VIS_HEADERS} "./Trackball Camera/*.h")
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("Include" FILES ${VIS_INCLUDE})
source_group("" FILES ${VIS_SOURCES} ${VIS_HEADERS}) 
source_group("shaders\\vertex" FILES "VertexShader.glsl")
source_group("shaders\\fragment" FILES "NodeFragmentShader.glsl" "EdgeFragmentShader.glsl")
source_group("Source Files" FILES	"Marker.h" "Marker.cpp"
									"MarkerHistogram.h" "MarkerHistogram.cpp"
									"MarkerGraph.h" "MarkerGraph.cpp") 
source_group("Source Files\\Common\\Color Spaces" FILES "colorspaces.h")
source_group("Source Files\\Common\\Trackball Camera" FILES "./Trackball Camera/TrackballCamera.h" "./Trackball Camera/TrackballCamera.cpp") 
source_group("Source Files\\Common\\Trackball Camera" FILES "./Trackball Camera/CameraControl.h" "./Trackball Camera/CameraControl.cpp") 

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${PROJECT_SOURCE_DIR}/modules
					${OpenCV_INCLUDE_DIRS} 
				)
  
# Properties -> Linker -> General -> Additional Library Directories
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})  
  
# Set Properties -> General -> Configuration Type to Dynamic Library(.dll)
add_library(VIS SHARED ${VIS_INCLUDE} ${VIS_SOURCES} ${VIS_HEADERS} ${VIS_SHADERS})
add_dependencies(VIS DGM)
 
# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(VIS PRIVATE ${OpenCV_LIBS} ${DGM_LIB})

if(USE_OPENGL)
	include_directories(${PROJECT_SOURCE_DIR}/3rdparty/glew-2.0.0/include
						${PROJECT_SOURCE_DIR}/3rdparty/glm-0.9.8.0
						"Trackball Camera")

	# GLEW
	add_library(glew STATIC ${PROJECT_SOURCE_DIR}/3rdparty/glew-2.0.0/src/glew.c)
	target_link_libraries(glew ${OPENGL_LIBRARY} ${EXTRA_LIBS})	
	set_target_properties(glew PROPERTIES FOLDER "3rdparty")
	add_definitions(-DGLEW_STATIC)
	
	add_dependencies(VIS glew glfw)
	
	target_link_libraries(VIS PRIVATE glew glfw ${OPENGL_gl_LIBRARY})	
endif()

set_target_properties(VIS PROPERTIES OUTPUT_NAME_RELEASE vis${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH})
set_target_properties(VIS PROPERTIES OUTPUT_NAME_DEBUG vis${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d)

#install
install(TARGETS VIS
	EXPORT DGMTargets
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${VIS_INCLUDE} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${VIS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/VIS)

# Creates folder "Modules" and adds target project 
set_target_properties(VIS PROPERTIES FOLDER "Modules")
 