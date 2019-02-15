# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include 
					${PROJECT_SOURCE_DIR}/modules
					${PROJECT_SOURCE_DIR}/3rdparty
					${OpenCV_INCLUDE_DIRS} 
				)

# Properties -> Linker -> General -> Additional Library Directories
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
				
# ================================================= DEMO 1D =================================================
add_executable(Demo_1D "Demo 1D.cpp" "Demo 1D.h" "chain.cpp" "chain.h" "exact.cpp" "exact.h" "tree.cpp" "tree.h")
add_dependencies(Demo_1D DGM VIS)

source_group("Source Files" FILES "Demo 1D.h" "Demo 1D.cpp")
source_group("Source Files\\Chain" FILES "chain.h" "chain.cpp")
source_group("Source Files\\Exact" FILES "exact.h" "exact.cpp")
source_group("Source Files\\Tree" FILES "tree.h" "tree.cpp")

set_target_properties(Demo_1D PROPERTIES PROJECT_LABEL "Demo 1D")						# in Visual Studio
set_target_properties(Demo_1D PROPERTIES OUTPUT_NAME "Demo 1D")
set_target_properties(Demo_1D PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_1D ${OpenCV_LIBS} ${DGM_LIB} ${VIS_LIB})

#install
install(TARGETS Demo_1D RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# ================================================= DEMO 2D =================================================
add_executable(Demo_2D "Demo 2D.cpp")
add_dependencies(Demo_2D DGM)

set_target_properties(Demo_2D PROPERTIES PROJECT_LABEL "Demo 2D")						# in Visual Studio
set_target_properties(Demo_2D PROPERTIES OUTPUT_NAME "Demo 2D")
set_target_properties(Demo_2D PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_2D ${OpenCV_LIBS} ${DGM_LIB})  

#install
install(TARGETS Demo_2D RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# ================================================= DEMO FEX ================================================
add_executable(Demo_FEX "Demo FEX.cpp")
add_dependencies(Demo_FEX FEX)

set_target_properties(Demo_FEX PROPERTIES PROJECT_LABEL "Demo Feature Extraction")		# in Visual Studio
set_target_properties(Demo_FEX PROPERTIES OUTPUT_NAME "Demo Feature Extraction")
set_target_properties(Demo_FEX PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_FEX ${OpenCV_LIBS} ${FEX_LIB})  

#install
install(TARGETS Demo_FEX RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# ============================================ DEMO RANDOM MODEL ============================================
add_executable(Demo_RandomModel "Demo Random Model.cpp")
add_dependencies(Demo_RandomModel DGM VIS FEX)

set_target_properties(Demo_RandomModel PROPERTIES PROJECT_LABEL "Demo Random Model")	# in Visual Studio
set_target_properties(Demo_RandomModel PROPERTIES OUTPUT_NAME "Demo Random Model")
set_target_properties(Demo_RandomModel PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_RandomModel ${OpenCV_LIBS} ${DGM_LIB} ${VIS_LIB} ${FEX_LIB})

#install
install(TARGETS Demo_RandomModel RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# =============================================== DEMO TRAIN ================================================
add_executable(Demo_Train "Demo Train.cpp")
add_dependencies(Demo_Train DGM VIS)

set_target_properties(Demo_Train PROPERTIES PROJECT_LABEL "Demo Train")					# in Visual Studio
set_target_properties(Demo_Train PROPERTIES OUTPUT_NAME "Demo Train")
set_target_properties(Demo_Train PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_Train ${OpenCV_LIBS} ${DGM_LIB} ${VIS_LIB})

#install
install(TARGETS Demo_Train RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# =============================================== DEMO DENSE ================================================
add_executable(Demo_Dense "Demo Dense.cpp")
add_dependencies(Demo_Dense DGM VIS)

set_target_properties(Demo_Dense PROPERTIES PROJECT_LABEL "Demo Dense")					# in Visual Studio
set_target_properties(Demo_Dense PROPERTIES OUTPUT_NAME "Demo Dense")
set_target_properties(Demo_Dense PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_Dense ${OpenCV_LIBS} ${DGM_LIB} ${VIS_LIB})

#install
install(TARGETS Demo_Dense RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# ========================================= DEMO PARAM ESTIMATION ===========================================
add_executable(Demo_ParamEstimation "Demo Param Estimation.cpp")
add_dependencies(Demo_ParamEstimation DGM VIS)

set_target_properties(Demo_ParamEstimation PROPERTIES PROJECT_LABEL "Demo Param Estimation")					# in Visual Studio
set_target_properties(Demo_ParamEstimation PROPERTIES OUTPUT_NAME "Demo Param Estimation")
set_target_properties(Demo_ParamEstimation PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_ParamEstimation ${OpenCV_LIBS} ${DGM_LIB} ${VIS_LIB})

#install
install(TARGETS Demo_ParamEstimation RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
	
# =========================================== DEMO VISUALIZATION ============================================
add_executable(Demo_Vis "Demo Visualization.cpp")
add_dependencies(Demo_Vis DGM VIS)

set_target_properties(Demo_Vis PROPERTIES PROJECT_LABEL "Demo Visualization")			# in Visual Studio
set_target_properties(Demo_Vis PROPERTIES OUTPUT_NAME "Demo Visualization")
set_target_properties(Demo_Vis PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_Vis ${OpenCV_LIBS} ${DGM_LIB} ${VIS_LIB})

#install
install(TARGETS Demo_Vis RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# ============================================== DEMO STEREO ===============================================
add_executable(Demo_Stereo "Demo Stereo.cpp")
add_dependencies(Demo_Stereo DGM)

set_target_properties(Demo_Stereo PROPERTIES PROJECT_LABEL "Demo Stereo")			# in Visual Studio
set_target_properties(Demo_Stereo PROPERTIES OUTPUT_NAME "Demo Stereo")
set_target_properties(Demo_Stereo PROPERTIES FOLDER "Demos")

# Properties->Linker->Input->Additional Dependencies
target_link_libraries(Demo_Stereo ${OpenCV_LIBS} ${DGM_LIB})

#install
install(TARGETS Demo_Stereo RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
