// Graph interface class;
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "Graph.h"

namespace DirectGraphicalModels {
	// ================================ Graph Interface Class ================================
	/**
	* @brief Interface class for graphical models
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class IGraphPairwise : public CGraph
	{
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport IGraphPairwise(byte nStates) : CGraph(nStates) {}
		DllExport virtual ~IGraphPairwise(void) {}


		/**
		* @brief Returns the set of IDs of the child nodes of the argument node
		* @param[in] node node index
		* @param[out] vNodes vector with the child node's ID
		*/
		DllExport virtual void		getChildNodes(size_t node, vec_size_t &vNodes) const = 0;
		/**
		* @brief Returns the set of IDs of the parent nodes of the argument node
		* @param[in] node node index
		* @param[out] vNodes vector with the parent node's ID
		*/
		DllExport virtual void		getParentNodes(size_t node, vec_size_t &vNodes) const = 0;
		/**
		* @brief Adds an additional directed edge
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		*/
		DllExport virtual void		addEdge(size_t srcNode, size_t dstNode) = 0;
		/**
		* @brief Adds an additional directed edge with specified potentional
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @param pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		addEdge(size_t srcNode, size_t dstNode, const Mat &pot) = 0;
		/**
		* @brief Sets or changes the potentional of directed edge
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @param pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		setEdge(size_t srcNode, size_t dstNode, const Mat &pot) = 0;
		/**
		* @brief Returns the edge potential
		* @param[in] srcNode index of the source node
		* @param[in] dstNode index of the destination node
		* @param[out] pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1) if exists, empty Mat otherwise
		*/
		DllExport virtual void		getEdge(size_t srcNode, size_t dstNode, Mat &pot) const = 0;
		/**
		* @brief Assigns a directed edge (\b srcNode) --> (\b dstNode) to the group \b group
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @param group The edge group ID
		*/
		DllExport virtual void		setEdgeGroup(size_t srcNode, size_t dstNode, byte group) = 0;
		/**
		* @brief Returns the group of the edge
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @returns The edge group ID
		*/
		DllExport virtual byte		getEdgeGroup(size_t srcNode, size_t dstNode) const = 0;
		/**
		* @brief Removes the specified edge
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		*/		
		DllExport virtual void		removeEdge(size_t srcNode, size_t dstNode) = 0;
		/**
		* @brief Checks whether the edge exists
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @retval true if the edge exists
		* @retval false otherwise
		*/
		DllExport virtual bool		isEdgeExists(size_t srcNode, size_t dstNode) const = 0;
		/**
		* @brief Checks whether the edge is a part of an arc
		* @details In contrast to the isArcExists() function, this function does not checks whether the input edge exists, and thus faster
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		* @retval true if the edge is a part of an arc
		* @retval false otherwise
		*/
		DllExport virtual bool		isEdgeArc(size_t srcNode, size_t dstNode) const = 0;
		/**
		* @brief Adds an additional udirected edge (arc)
		* @details The arc is emulated by adding two directed edges
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		*/
		DllExport virtual void		addArc(size_t Node1, size_t Node2) = 0;
		/**
		* @brief Adds an additional udirected edge (arc) with specified potentional
		* @details The arc is emulated by adding two directed edges. For sake of consistency the pot matrix here is squarerooted:
		* @code
		* addEdge(Node1, Node2, sqrt(pot));
		* addEdge(Node2, Node1, sqrt(pot));
		* @endcode
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		* @param pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		addArc(size_t Node1, size_t Node2, const Mat &pot) = 0;
		/**
		* @brief Sets or changes the potentional of udirected edge (arc)
		* @details The arc is emulated by adding two directed edges. For sake of consistency the pot matrix here is squarerooted:
		* @code
		* addEdge(Node1, Node2, sqrt(pot));
		* addEdge(Node2, Node1, sqrt(pot));
		* @endcode
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		* @param pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		setArc(size_t Node1, size_t Node2, const Mat &pot) = 0;
		/**
		* @brief Assigns an undirected edge (arc) (\b Node1) -- (\b Node2) to the group \b group
		* @param Node1 index of the source node
		* @param Node2 index of the destination node
		* @param group The edge group ID
		*/
		DllExport virtual void		setArcGroup(size_t Node1, size_t Node2, byte group) = 0;
		/**
		* @brief Removes the specified arc
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		*/
		DllExport virtual void		removeArc(size_t Node1, size_t Node2) = 0;
		/**
		* @brief Checks whether the arc exists
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		* @retval true if the arc exists
		* @retval false otherwise
		*/
		DllExport virtual bool		isArcExists(size_t Node1, size_t Node2) const = 0;
	};
}  
