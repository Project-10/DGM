// Graph Marker Class
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "Marker.h"

#define DGM_HSV(h, s, v) cvScalar(h, s, v, 0 )

namespace DirectGraphicalModels
{
	class CBaseGraph;
	// ================================ Histogram Marker Class ================================
	/**
	* @ingroup moduleVis
	* @brief Graph Marker class
	* @details This class allows to visualize a graph, described by CGraph class.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CMarkerGraph : public CMarker
	{
	public:
		/**
		* @brief Constructor
		* @param pGraph The graph
		*/
		DllExport CMarkerGraph(CBaseGraph *pGraph) : CMarker(), m_pGraph(pGraph) {}
		DllExport virtual ~CMarkerGraph(void) {}

		/**
		* @brief Visualizes the graph structure
		* @param posFunc The pointer to a positioning function. Refere to @ref drawGraph() function for more details.
		* @return Image 1000 x 1000 pixels with visualized graph.
		*/
		DllExport inline Mat drawGraph(CvPoint(*posFunc) (size_t nodeId, int size)) const { return drawGraph(m_pGraph, posFunc); }
		/**
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
		DllExport static Mat drawGraph(CBaseGraph *pGraph, CvPoint (*posFunc) (size_t nodeId, int size) );

	
	protected:
		CBaseGraph			* m_pGraph;			///< Pointer to the graph


	private:
		static CvScalar		  hsv2rgb(CvScalar hsv);


	private:
		static const byte	  bkgIntencity;
	};
}