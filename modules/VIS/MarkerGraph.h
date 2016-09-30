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
	* @param size The size of resulting image
	* @param pGraph The graph
	* @param posFunc The pointer to a positioning function: a mapper, which places every node at the resulting image canvas of size: \a size x \a size.
	* For example:
	* @code
	* CvPoint posFunc(size_t nodeId, int size) {
	*	return Point2f(
	*		0.5f + 0.45 * cos(2 * n * Pi / nNodes),
	*		0.5f + 0.45 * sin(2 * n * Pi / nNodes) );
	*	});
	* @endcode
	* @return Image \b size x \b size pixels with visualized graph.
	*/
	DllExport Mat drawGraph(int size, IGraph *pGraph, Point2f (*posFunc) (size_t nodeId) );

#ifdef USE_OPENGL
	/**
	* @ingroup moduleVIS
	* @brief
	* @details
	* > In order to use this function, OpenGL must be build with the \b USE_OPENGL flag
	*/
	DllExport void drawGraph3D(int size, IGraph *pGraph, Point3f(*posFunc) (size_t nodeId));
#endif
} }