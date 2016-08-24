file(GLOB MARKER_INCLUDE	${PROJECT_SOURCE_DIR}/include/Marker.h)
file(GLOB MARKER_SOURCES	"*.cpp")
file(GLOB MARKER_HEADERS	"*.h")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("Include" FILES ${MARKER_INCLUDE})
#source_group("" FILES ${MARKER_SOURCES} ${MARKER_HEADERS}) 
source_group("Source Files" FILES ${MARKER_SOURCES} ${MARKER_HEADERS}) 
#source_group("Source Files\\Common" FILES "SquareNeighborhood.h")
#source_group("Source Files\\Feature Extractor" FILES "BaseFeatureExtractor.h" "BaseFeatureExtractor.cpp" "CommonFeatureExtractor.h")
#source_group("Source Files\\Feature Extractor\\Coordinate" FILES "Coordinate.h" "Coordinate.cpp")
#source_group("Source Files\\Feature Extractor\\Distance" FILES "Distance.h" "Distance.cpp")
#source_group("Source Files\\Feature Extractor\\Gradient" FILES "Gradient.h" "Gradient.cpp")
#source_group("Source Files\\Feature Extractor\\HOG" FILES "HOG.h" "HOG.cpp")
#source_group("Source Files\\Feature Extractor\\HSV" FILES "HSV.h" "HSV.cpp")
#source_group("Source Files\\Feature Extractor\\Intensity" FILES "Intensity.h" "Intensity.cpp")
#source_group("Source Files\\Feature Extractor\\NDVI" FILES "NDVI.h" "NDVI.cpp")
#source_group("Source Files\\Feature Extractor\\Scale" FILES "Scale.h" "Scale.cpp")
#source_group("Source Files\\Feature Extractor\\Sparse Coding" FILES "SparseCoding.h" "SparseCoding.cpp" "SparseDictionary.h" "SparseDictionary.cpp")
#source_group("Source Files\\Feature Extractor\\Variance" FILES "Variance.h" "Variance.cpp")

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${PROJECT_SOURCE_DIR}/modules
					${OpenCV_INCLUDE_DIRS} 
					${OpenCV_INCLUDE_DIRS}2
				)
  
# Properties -> Linker -> General -> Additional Library Directories
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})  
  
# Set Properties -> General -> Configuration Type to Dynamic Library(.dll)
add_library(Marker SHARED ${MARKER_INCLUDE} ${MARKER_SOURCES} ${MARKER_HEADERS})
add_dependencies(Marker DGM)
 
# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(Marker ${OpenCV_LIBS} ${DGM_LIB})
 
set_target_properties(Marker PROPERTIES OUTPUT_NAME_RELEASE marker${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH})
set_target_properties(Marker PROPERTIES OUTPUT_NAME_DEBUG marker${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d)

#install
install(TARGETS Marker 
	RUNTIME DESTINATION ${PROJECT_INSTALL_PREFIX}/bin 
	LIBRARY DESTINATION ${PROJECT_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${PROJECT_INSTALL_PREFIX}/lib)
install(FILES ${MARKER_INCLUDE} DESTINATION ${PROJECT_INSTALL_PREFIX}/include)
install(FILES ${MARKER_HEADERS} DESTINATION ${PROJECT_INSTALL_PREFIX}/include/Marker)

# Creates folder "Modules" and adds target project 
set_target_properties(Marker PROPERTIES FOLDER "Modules")
 
