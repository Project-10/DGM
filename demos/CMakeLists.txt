# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include 
					${PROJECT_SOURCE_DIR}/modules
					${PROJECT_SOURCE_DIR}/3rdparty
					${OpenCV_INCLUDE_DIRS} 
				)

# Properties -> Linker -> General -> Additional Library Directories
link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

function(create_demo target source dependencies)
	add_executable(${target} "${source}.cpp")
	
	set_target_properties(${target} PROPERTIES PROJECT_LABEL ${source})			# in Visual Studio
	set_target_properties(${target} PROPERTIES OUTPUT_NAME ${source})
	set_target_properties(${target} PROPERTIES FOLDER "Demos")
	
	# Properties->Linker->Input->Additional Dependencies
	target_link_libraries(${target} ${OpenCV_LIBS})
	foreach(dependency IN LISTS dependencies)
		add_dependencies(${target} ${dependency})
		target_link_libraries(${target} ${${dependency}_LIB})
	endforeach()

	#install
	install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)	
endfunction(create_demo)

				
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
create_demo(Demo_2D "Demo 2D" "DGM")

# ================================================= DEMO FEX ================================================
create_demo(Demo_FEX "Demo FEX" "FEX")

# ============================================ DEMO RANDOM MODEL ============================================
create_demo(Demo_RandomModel "Demo Random Model" "DGM;VIS;FEX")

# =============================================== DEMO TRAIN ================================================
create_demo(Demo_Train "Demo Train" "DGM;VIS")

# =============================================== DEMO DENSE ================================================
create_demo(Demo_Dense "Demo Dense" "DGM;VIS")

# ========================================= DEMO PARAM ESTIMATION ===========================================
create_demo(Demo_ParamEstimation "Demo Param Estimation" "DGM;VIS")
	
# =========================================== DEMO VISUALIZATION ============================================
create_demo(Demo_Vis "Demo Visualization" "DGM;VIS")

# ============================================== DEMO STEREO ================================================
create_demo(Demo_Stereo "Demo Stereo" "DGM")

# ================================================ DEMO ICR =================================================
create_demo(Demo_ICR "Demo ICR" "DGM;DNN;VIS")

# ================================================ DEMO RBM =================================================
create_demo(Demo_RBM "Demo RBM" "DGM;DNN;VIS")
