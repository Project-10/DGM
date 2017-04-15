file(GLOB FEX_INCLUDE	${PROJECT_SOURCE_DIR}/include/FEX.h)
file(GLOB FEX_SOURCES	"*.cpp")
file(GLOB FEX_HEADERS	"*.h")

file(GLOB 3RD_OPENCV_SOURCES	"${PROJECT_SOURCE_DIR}/3rdparty/opencv/SIFT.h"
								"${PROJECT_SOURCE_DIR}/3rdparty/opencv/SIFT.cpp"
							)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("Include" FILES ${FEX_INCLUDE})
source_group("" FILES ${FEX_SOURCES} ${FEX_HEADERS}) 
source_group("3rdparty\\opencv" FILES ${3RD_OPENCV_SOURCES})
source_group("Source Files\\Common\\Linear Mapper" FILES "LinearMapper.h")
source_group("Source Files\\Common\\Square Neighborhood" FILES "SquareNeighborhood.h")
source_group("Source Files\\Feature Extractor" FILES "IFeatureExtractor.h")
source_group("Source Files\\Feature Extractor\\Common Feature Extractor" FILES "CommonFeatureExtractor.h" "CommonFeatureExtractor.cpp")
source_group("Source Files\\Feature Extractor\\Local" FILES "ILocalFeatureExtractor.h")
source_group("Source Files\\Feature Extractor\\Local\\Coordinate" FILES "Coordinate.h" "Coordinate.cpp")
source_group("Source Files\\Feature Extractor\\Local\\Distance" FILES "Distance.h" "Distance.cpp")
source_group("Source Files\\Feature Extractor\\Local\\Gradient" FILES "Gradient.h" "Gradient.cpp")
source_group("Source Files\\Feature Extractor\\Local\\HOG" FILES "HOG.h" "HOG.cpp")
source_group("Source Files\\Feature Extractor\\Local\\SIFT" FILES "SIFT.h" "SIFT.cpp")
source_group("Source Files\\Feature Extractor\\Local\\HSV" FILES "HSV.h" "HSV.cpp")
source_group("Source Files\\Feature Extractor\\Local\\Intensity" FILES "Intensity.h" "Intensity.cpp")
source_group("Source Files\\Feature Extractor\\Local\\NDVI" FILES "NDVI.h" "NDVI.cpp")
source_group("Source Files\\Feature Extractor\\Local\\Scale" FILES "Scale.h" "Scale.cpp")
source_group("Source Files\\Feature Extractor\\Local\\Sparse Coding" FILES "SparseCoding.h" "SparseCoding.cpp" "SparseDictionary.h" "SparseDictionary.cpp")
source_group("Source Files\\Feature Extractor\\Local\\Variance" FILES "Variance.h" "Variance.cpp")
source_group("Source Files\\Feature Extractor\\Global" FILES "GlobalFeatureExtractor.h" "Global.h" "Global.cpp")

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${PROJECT_SOURCE_DIR}/modules
					${PROJECT_SOURCE_DIR}/3rdparty
					${OpenCV_INCLUDE_DIRS} 
				)
  
# Set Properties -> General -> Configuration Type to Dynamic Library(.dll)
add_library(FEX SHARED ${FEX_INCLUDE} ${FEX_SOURCES} ${FEX_HEADERS})
 
# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(FEX ${OpenCV_LIBS})
 
set_target_properties(FEX PROPERTIES OUTPUT_NAME_RELEASE fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH})
set_target_properties(FEX PROPERTIES OUTPUT_NAME_DEBUG fex${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d)

#install
install(TARGETS FEX 
	EXPORT DGMTargets
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${FEX_INCLUDE} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${FEX_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/FEX)

# 3rdparty code
target_sources(FEX PRIVATE	${3RD_OPENCV_SOURCES} )

# Creates folder "Modules" and adds target project 
set_target_properties(FEX PROPERTIES FOLDER "Modules")
 
