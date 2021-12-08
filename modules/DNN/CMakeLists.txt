file(GLOB DNN_INCLUDE	${PROJECT_SOURCE_DIR}/include/DNN.h)
file(GLOB DNN_SOURCES	"*.cpp")
file(GLOB DNN_HEADERS	"*.h")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("Include" FILES ${DNN_INCLUDE})
source_group("" FILES ${DNN_SOURCES} ${DNN_HEADERS}) 
source_group("Source Files\\Neuron" FILES	"Neuron.h" "Neuron.cpp")
source_group("Source Files\\Neuron Layer" FILES	"NeuronLayer.h" "NeuronLayer.cpp")
source_group("Source Files\\Perceptron" FILES	"Perceptron.h" "Perceptron.cpp")
source_group("Source Files\\RBM" FILES	"RBM.h" "RBM.cpp")

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${PROJECT_SOURCE_DIR}/modules
					${OpenCV_INCLUDE_DIRS} 
				)
  
# Properties -> Linker -> General -> Additional Library Directories
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})  
  
# Set Properties -> General -> Configuration Type to Dynamic Library(.dll)
add_library(DNN SHARED ${DNN_INCLUDE} ${DNN_SOURCES} ${DNN_HEADERS} ${DNN_SHADERS})
add_dependencies(DNN DGM)
 
# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(DNN PRIVATE ${OpenCV_LIBS} ${DGM_LIB})

set_target_properties(DNN PROPERTIES OUTPUT_NAME dnn${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH})
set_target_properties(DNN PROPERTIES VERSION ${DGM_VERSION_MAJOR}.${DGM_VERSION_MINOR}.${DGM_VERSION_PATCH} SOVERSION ${DGM_VERSION_MAJOR}.${DGM_VERSION_MINOR}.${DGM_VERSION_PATCH})

#install
install(TARGETS DNN
	EXPORT DGMTargets
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${DNN_INCLUDE} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${DNN_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/DNN)

# Creates folder "Modules" and adds target project 
set_target_properties(DNN PROPERTIES FOLDER "Modules")
 
