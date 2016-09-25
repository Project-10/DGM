// Graph Marker Class
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels { 
	class IGraph;
	namespace vis
{
	/**
	* @ingroup moduleVIS
	* @brief Visualizes the graph structure
	* @param pGraph The graph
	* @param posFunc The pointer to a positioning function: a mapper, which places every node at the resulting image canvas of size: \a size x \a size.
	* For example:
	* @code
	* CvPoint posFunc(size_t nodeId, int size) {
	*	return cvPoint(
	*		size / 2 + static_cast<int>(0.45 * size * cos(2 * nodeId * Pi / nNodes)),
	*		size / 2 + static_cast<int>(0.45 * size * sin(2 * nodeId * Pi / nNodes)) );
	* }
	* @endcode
	* @return Image 1000 x 1000 pixels with visualized graph.
	*/
	DllExport Mat drawGraph(IGraph *pGraph, CvPoint (*posFunc) (size_t nodeId, int size) );
} }