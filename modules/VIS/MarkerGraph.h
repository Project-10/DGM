// Graph Marker Class
// Written by Sergey Kosov in 2015 - 2016 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels { 
	class IGraphPairwise;
	namespace vis
{
	/**
	* @ingroup moduleVIS
	* @brief Visualizes the graph structure
	* @param size The size of resulting image
	* @param graph The graph
	* @param posFunc The positioning function: a mapper, that defines spacial position at the canvas for every graph node.
	* The coordinates must lie in range [-1; 1].<br>
	* For example:
	* @code
	* Point2f posFunc(size_t nodeId) {
	*	return Point2f(
	*		0.9f * cosf(2 * n * Pif / nNodes),
	*		0.9f * sinf(2 * n * Pif / nNodes) 
	*	);
	* });
	* @endcode
	* @param colorFunc The color function: a mapper, that defines color (\b CV_RGB(r, g, b)) for every graph node.
	* @param groupsColor The list of colors for graph edge groups. May be achieved with the function:
	* \ref vis::generateDefaultPalette() or generated manually.
	* @return Image \b size x \b size pixels with visualized graph.
	*/
	DllExport Mat drawGraph(int								  size, 
							IGraphPairwise					& graph,
							std::function<Point2f(size_t)>	  posFunc, 
							std::function<Scalar(size_t)>	  colorFunc	= nullptr, 
							const vec_scalar_t				& groupsColor	= vec_scalar_t());

#ifdef USE_OPENGL
	/**
	* @ingroup moduleVIS
	* @brief Visualizes the graph structure in 3D
	* @details This function creates an OpenGL window with the visualized graph, seen from a trackball camera.
	* User may rotate, zoom, pan and centralize the visualized graph.
	* > In order to use this function, OpenGL must be built with the \b USE_OPENGL flag
	* @param size The size of the viewing window (\b size x \b size pixels)
	* @param graph The graph
	* @param posFunc The positioning function: a mapper, that defines spacial position in 3D world for every graph node.
	* For better wieving expierence, the coordinates should lie in range [-0.5; 0.5].<br>
	* For example:
	* @code
	* Point3f posFunc(size_t nodeId) {
	*	return Point3f(
	*		0.9f * cosf(2 * n * Pif / nNodes),
	*		0.9f * sinf(2 * n * Pif / nNodes),
	*		static_cast<float>(n) / nNodes
	*	);
	* });
	* @endcode
	* @param colorFunc The color function: a mapper, that defines color (\b CV_RGB(r, g, b)) for every graph node.
	* @param groupsColor The list of colors for graph edge groups. May be achieved with the function:
	* \ref vis::generateDefaultPalette() or generated manually.                                 
	*/
	DllExport void showGraph3D(int								 size, 
							   IGraphPairwise				   & graph,
							   std::function<Point3f(size_t)>    posFunc, 
							   std::function<cv::Scalar(size_t)>   colorFunc	= nullptr,
							   const vec_scalar_t			   & groupsColor	= vec_scalar_t());
#endif
} }
