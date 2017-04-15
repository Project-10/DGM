file(GLOB DGM_INCLUDE	${PROJECT_SOURCE_DIR}/include/DGM.h
						${PROJECT_SOURCE_DIR}/include/types.h
						${PROJECT_SOURCE_DIR}/include/macroses.h
						)
file(GLOB DGM_SOURCES	"*.cpp")
file(GLOB DGM_HEADERS	"*.h")
					
file(GLOB 3RD_OPENCV_SOURCES	"${PROJECT_SOURCE_DIR}/3rdparty/opencv/DTrees.h" "${PROJECT_SOURCE_DIR}/3rdparty/opencv/RTrees.h"
								"${PROJECT_SOURCE_DIR}/3rdparty/opencv/DTrees.cpp" "${PROJECT_SOURCE_DIR}/3rdparty/opencv/RTrees.cpp"
							)
							
if (USE_SHERWOOD)
file(GLOB_RECURSE 3RD_SHERWOOD_SOURCES	"${PROJECT_SOURCE_DIR}/3rdparty/sherwood/*.h"
										"${PROJECT_SOURCE_DIR}/3rdparty/sherwood/*.cpp"
									)
endif()
 
# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("Include" FILES ${DGM_INCLUDE})
source_group("" FILES ${DGM_SOURCES} ${DGM_HEADERS})
source_group("3rdparty\\opencv" FILES ${3RD_OPENCV_SOURCES})
source_group("3rdparty\\sherwood" FILES ${3RD_SHERWOOD_SOURCES})
source_group("Source Files\\Common\\Confusion Matrix" FILES "CMat.h" "CMat.cpp")
source_group("Source Files\\Common\\Features Concatenator" FILES "FeaturesConcatenator.h")
source_group("Source Files\\Common\\Average Precision" FILES "AveragePrecision.h" "AveragePrecision.cpp")
source_group("Source Files\\Common\\KDGauss" FILES "NDGauss.h" "NDGauss.cpp")
source_group("Source Files\\Common\\KDTree" FILES "KDTree.h" "KDTree.cpp" "KDNode.h" "KDNode.cpp")
source_group("Source Files\\Common\\Samples Accumulator" FILES "SamplesAccumulator.h" "SamplesAccumulator.cpp")
source_group("Source Files\\Common\\Random" FILES "random.h")
source_group("Source Files\\Common\\Math Operations" FILES "mathop.h")
source_group("Source Files\\Common\\Parallel" FILES "parallel.h")
source_group("Source Files\\Decoding" FILES "Decode.h" "Decode.cpp")												
source_group("Source Files\\Decoding\\Exact" FILES "DecodeExact.h" "DecodeExact.cpp")												
source_group("Source Files\\Graph" FILES	"IGraph.h"
											"Graph.h"
											"Graph.cpp"
											"GraphWeiss.h"
											"GraphWeiss.cpp"
											"Graph3.h"
											"Graph3.cpp"
											"GraphBoost.h"
											"GraphBoost.cpp"
											"GraphExt.h"
											"GraphLayered.h"
											"GraphLayered.cpp"
									)
source_group("Source Files\\Inference" FILES "Infer.h" "Infer.cpp")
source_group("Source Files\\Inference\\Exact" FILES "InferExact.h" "InferExact.cpp")
source_group("Source Files\\Inference\\Message Passing" FILES "MessagePassing.h" "MessagePassing.cpp")
source_group("Source Files\\Inference\\Message Passing\\Chain" FILES "InferChain.h" "InferChain.cpp")
source_group("Source Files\\Inference\\Message Passing\\LBP" FILES "InferLBP.h" "InferLBP.cpp")
source_group("Source Files\\Inference\\Message Passing\\Tree" FILES "InferTree.h" "InferTree.cpp")
source_group("Source Files\\Inference\\Message Passing\\TRW" FILES "InferTRW.h" "InferTRW.cpp")
source_group("Source Files\\Inference\\Message Passing\\Viterbi" FILES "InferViterbi.h")
source_group("Source Files\\Param Estimation" FILES "Powell.h" "Powell.cpp")
source_group("Source Files\\Random Model" FILES "BaseRandomModel.h" "BaseRandomModel.cpp")
source_group("Source Files\\Random Model\\PDF" FILES "IPDF.h")
source_group("Source Files\\Random Model\\PDF\\Gaussian" FILES "PDFGaussian.h" "PDFGaussian.cpp")
source_group("Source Files\\Random Model\\PDF\\Histogram" FILES "PDFHistogram.h" "PDFHistogram.cpp")
source_group("Source Files\\Random Model\\Prior" FILES "Prior.h" "Prior.cpp")
source_group("Source Files\\Random Model\\Prior\\Edge" FILES "PriorEdge.h" "PriorEdge.cpp")
source_group("Source Files\\Random Model\\Prior\\Node" FILES "PriorNode.h" "PriorNode.cpp")
source_group("Source Files\\Random Model\\Prior\\Triplet" FILES "PriorTriplet.h" "PriorTriplet.cpp")
source_group("Source Files\\Random Model\\Training" FILES "ITrain.h")
source_group("Source Files\\Random Model\\Training\\Edge" FILES "TrainEdge.h" "TrainEdge.cpp")
source_group("Source Files\\Random Model\\Training\\Edge\\Contrast Sensitive" FILES "TrainEdgePottsCS.h" "TrainEdgePottsCS.cpp")
source_group("Source Files\\Random Model\\Training\\Edge\\Data Dependent" FILES "TrainEdgePrior.h" "TrainEdgePrior.cpp" "TrainEdgeConcat.h")
source_group("Source Files\\Random Model\\Training\\Edge\\Potts" FILES "TrainEdgePotts.h" "TrainEdgePotts.cpp")
source_group("Source Files\\Random Model\\Training\\Link" FILES "TrainLink.h" "TrainLink.cpp" "TrainLinkNested.h")
source_group("Source Files\\Random Model\\Training\\Node" FILES "TrainNode.h" "TrainNode.cpp")
source_group("Source Files\\Random Model\\Training\\Node\\Gaussian Mixture" FILES	"TrainNodeCvGM.h" 
																					"TrainNodeCvGMM.h"
																					"TrainNodeCvGMM.cpp"
																					"TrainNodeGM.h" 
																					"TrainNodeGMM.h"
																					"TrainNodeGMM.cpp"
																				)
source_group("Source Files\\Random Model\\Training\\Node\\Naive Bayes" FILES "TrainNodeNaiveBayes.h" "TrainNodeNaiveBayes.cpp")
source_group("Source Files\\Random Model\\Training\\Node\\K-Nearest Neighbors" FILES "TrainNodeKNN.h" "TrainNodeKNN.cpp")
source_group("Source Files\\Random Model\\Training\\Node\\Random Forest" FILES	"RForest.h" 
																				"RForest.cpp"
																				"TrainNodeCvRF.h"
																				"TrainNodeCvRF.cpp"
																				"TrainNodeMsRF.h"
																				"TrainNodeMsRF.cpp"
																			)
source_group("Source Files\\Random Model\\Training\\Triplet" FILES "TrainTriplet.h" "TrainTriplet.cpp") 
 
 
 
# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${PROJECT_SOURCE_DIR}/include
					${PROJECT_SOURCE_DIR}/3rdparty
					${OpenCV_INCLUDE_DIRS} 
				)
 
# Set Properties -> General -> Configuration Type to Dynamic Library(.dll)
add_library(DGM SHARED ${DGM_INCLUDE} ${DGM_SOURCES} ${DGM_HEADERS})
 
# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(DGM ${OpenCV_LIBS})

set_target_properties(DGM PROPERTIES OUTPUT_NAME_RELEASE dgm${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH})
set_target_properties(DGM PROPERTIES OUTPUT_NAME_DEBUG dgm${DGM_VERSION_MAJOR}${DGM_VERSION_MINOR}${DGM_VERSION_PATCH}d)



#install
install(TARGETS DGM 
	EXPORT DGMTargets
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin 
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${DGM_INCLUDE} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES ${DGM_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/DGM)
	
# 3rdparty code
target_sources(DGM PRIVATE	${3RD_OPENCV_SOURCES}
							${3RD_SHERWOOD_SOURCES}
						)
 
# Creates folder "Modules" and adds target project 
set_target_properties(DGM PROPERTIES FOLDER "Modules")


