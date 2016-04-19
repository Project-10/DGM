// Basic graph interface class;
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels {
	class CBaseGraph
	{
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CBaseGraph(byte nStates) : m_nStates(nStates) {};
		DllExport virtual ~CBaseGraph(void) {};

		/**
		* @brief Resets the graph
		* @details This function allows to re-use the graph or update node potential, while preserving the graph structure.
		* It deletes all the nodes and edges and sets node index to zero.
		*/
		DllExport virtual void		reset(void) = 0;
		/**
		* @brief Adds an additional node
		* @return the node's ID
		*/
		DllExport virtual size_t	addNode(void) = 0;
		/**
		* @brief Adds an additional node with specified potentional
		* @param pot node potential vector: Mat(size: nStates x 1; type: CV_32FC1)
		* @return the node's ID
		*/
		DllExport virtual size_t	addNode(const Mat &pot) = 0;
		/**
		* @brief Sets or changes the potential of node
		* @param node node index
		* @param pot node potential vector: Mat(size: nStates x 1; type: CV_32FC1)
		*/
		DllExport virtual void		setNode(size_t node, const Mat &pot) = 0;
		/**
		* @brief Returns the node potential
		* @param[in] node node index
		* @param[out] pot node potential vector: Mat(size: nStates x 1; type: CV_32FC1)
		*/
		DllExport virtual void		getNode(size_t node, Mat &pot) const = 0;
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
		* @brief Removes the specified edge
		* @param srcNode index of the source node
		* @param dstNode index of the destination node
		*/		
		DllExport virtual void		removeEdge(size_t srcNode, size_t dstNode) = 0;
		/**
		* @brief Adds an additional udirected edge (ark)
		* @details The ark is emulated by adding two directed edges
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		*/
		DllExport virtual void		addArk(size_t Node1, size_t Node2) = 0;
		/**
		* @brief Adds an additional udirected edge (ark) with specified potentional
		* @details The ark is emulated by adding two directed edges. For sake of consistency the pot matrix here is squarerooted:
		* @code
		* addEdge(Node1, Node2, sqrt(pot));
		* addEdge(Node2, Node1, sqrt(pot));
		* @endcode
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		* @param pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		addArk(size_t Node1, size_t Node2, const Mat &pot) = 0;
		/**
		* @brief Sets or changes the potentional of udirected edge (ark)
		* @details The ark is emulated by adding two directed edges. For sake of consistency the pot matrix here is squarerooted:
		* @code
		* addEdge(Node1, Node2, sqrt(pot));
		* addEdge(Node2, Node1, sqrt(pot));
		* @endcode
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		* @param pot edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		setArk(size_t Node1, size_t Node2, const Mat &pot) = 0;
		/**
		* @brief Removes the specified arc
		* @param Node1 index of the first node
		* @param Node2 index of the second node
		*/
		DllExport virtual void		removeArk(size_t Node1, size_t Node2) = 0;
		/**
		* @brief Returns the number of nodes in the graph
		* @returns number of nodes
		*/
		DllExport virtual size_t	getNumNodes(void) const = 0;
		/**
		* @brief Returns the number of edges in the graph
		* @returns number of edges
		*/
		DllExport virtual size_t	getNumEdges(void) const = 0;


	protected:
		byte m_nStates;		///< The number of states (classes)

	
	private:
		// Copy semantics are disabled
		CBaseGraph(const CBaseGraph &rhs) {}
		const CBaseGraph & operator= (const CBaseGraph & rhs) { return *this; }
	};
}  
