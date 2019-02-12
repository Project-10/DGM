// Dense Graph Edge Models interface class
// Written by Sergey G. Kosov in 2019 for Project X 
#pragma once

#include "types.h"

namespace DirectGraphicalModels {
    // ================================ Edge Models for Dense Graphs ================================
	/**
	* @brief Interface class for edge models used in dense graphical models
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class IEdgeModel {
	public:
		IEdgeModel(void) = default;
		IEdgeModel(const IEdgeModel&) = delete;
		virtual ~IEdgeModel(void) = default;
		
		const IEdgeModel& operator=(const IEdgeModel&) = delete;

		/**
		* @brief Applies an edge model to the node potentials of a dense graph.
		* @details This function subsequently (in terms of multiple iterations and multiple models) applies an edge 
		* model derived from this class to the node potentials provided via \b src argument and stores the result 
		* into the \b dst.
		* @param[in] src The dense graph node potentials in form Mat(size: nNodes x nStates; type: CV_32FC1)
		* @param[out] dst The reference to the container for resulting node potentials. Resulting matrix 
		* will be the same size and type as the input one: Mat(size: nNodes x nStates; type: CV_32FC1)
		*/
		virtual void apply(const Mat &src, Mat &dst) const = 0;
	};
}