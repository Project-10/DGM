file(GLOB FEX_INCLUDE	${PROJECT_SOURCE_DIR}/include/FEX.h)
file(GLOB FEX_SOURCES	"*.cpp")
file(GLOB FEX_HEADERS	"*.h")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("Include" FILES ${FEX_INCLUDE})
source_group("" FILES ${FEX_SOURCES} ${FEX_HEADERS}) 
source_group("Source Files\\Common" FILES "SquareNeighborhood.h" "LinearMapper.h")
source_group("Source Files\\Feature Extractor" FILES "IFeatureExtractor.h" "CommonFeatureExtractor.h")
source_group("Source Files\\Feature Extractor\\Coordinate" FILES "Coordinate.h" "Coordinate.cpp")
source_group("Source Files\\Feature Extractor\\Distance" FILES "Distance.h" "Distance.cpp")
source_group("Source Files\\Feature Extractor\\Gradient" FILES "Gradient.h" "Gradient.cpp")
source_group("Source Files\\Feature Extractor\\HOG" FILES "HOG.h" "HOG.cpp")
source_group("Source Files\\Feature Extractor\\HSV" FILES "HSV.h" "HSV.cpp")
source_group("Source Files\\Feature Extractor\\Intensity" FILES "Intensity.h" "Intensity.cpp")
source_group("Source Files\\Feature Extractor\\NDVI" FILES "NDVI.h" "NDVI.cpp")
source_group("Source Files\\Feature Extractor\\Scale" FILES "Scale.h" "Scale.cpp")
source_group("Source Files\\Feature Extractor\\Sparse Coding" FILES "SparseCoding.h" "SparseCoding.cpp" "SparseDictionary.h" "SparseDictionary.cpp")
source_group("Source Files\\Feature Extractor\\Variance" FILES "Variance.h" "Variance.cpp")

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${PROJECT_SOURCE_DIR}/modules
					${OpenCV_INCLUDE_DIRS} 
					${OpenCV_INCLUDE_DIRS}2
				)
  
# Set Properties -> General -> Configuration Type to Dynamic Library(.dll)
add_library(FEX SHARED ${FEX_INCLUDE} ${FEX_SOURCES} ${FEX_HEADERS})
 
# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(FEX ${OpenCV_LIBS})
 
set_target_properties(FEX PROPERTIES OUTPUT_NAME_RELEASE fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH})
set_target_properties(FEX PROPERTIES OUTPUT_NAME_DEBUG fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d)

#install
install(TARGETS FEX 
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${FEX_INCLUDE} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${FEX_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/FEX)

# Creates folder "Modules" and adds target project 
set_target_properties(FEX PROPERTIES FOLDER "Modules")
 
