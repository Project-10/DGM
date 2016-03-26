// (triplet) Graph class interface;                                             
// Written by Sergey G. Kosov in 2014 for Project X 
#pragma once

#include "Graph.h"

namespace DirectGraphicalModels
{
// =============================== Triplet Structure ==============================
/**
@brief %Triplet structure
@details Basic item stored in adjacency list. 
*/
	struct Triplet {
		size_t	node1;			///< First node in edge
		size_t	node2;			///< Second node in edge
		size_t	node3;			///< Third node in edge

		Triplet(void) {}
		
		Triplet(size_t n1, size_t n2, size_t n3) : node1(n1), node2(n2), node3(n3) {}
	}; 

	// ================================ Graph3 Class ================================
	/**
	* @brief Triple graph class
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraph3 : public CGraph
	{
	public:
/**
@brief Constructor
@param nStates the number of States (classes)
*/	
		DllExport CGraph3(byte nStates) : CGraph(nStates) {}
		DllExport virtual ~CGraph3(void) {}



/**
@brief Adds an additional directed edge 
@param[in] Node1 index of the first node
@param[in] Node2 index of the second node
@param[in] Node3 index of the third node
*/
		DllExport void		addTriplet(dword Node1, dword Node2, dword Node3);
/**
@brief Adds an additional directed edge with specified potentional
@param[in] Node1 index of the first node
@param[in] Node2 index of the second node
@param[in] Node3 index of the third node
@param[in] pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
*/
		DllExport void		addTriplet(dword Node1, dword Node2, dword Node3, const Mat &pot);
/**
@brief Sets or changes the potentional of directed edge
@param[in] Node1 index of the first node
@param[in] Node2 index of the second node
@param[in] Node3 index of the third node
@param[in] pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
*/
		DllExport void		setTriplet(dword Node1, dword Node2, dword Node3, const Mat &pot);

	protected:


	private:

	};
}