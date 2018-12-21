// Graph interface class;
// Written by Sergey Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels {
	// ================================ Graph Interface Class ================================
	/**
	* @brief Interface class for graphical models
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CGraph
	{
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport CGraph(byte nStates) : m_nStates(nStates) {};
		DllExport CGraph(const CGraph&) = delete;
		DllExport virtual ~CGraph(void) = default;

		const CGraph& operator= (const CGraph&) = delete;

		/**
		* @brief Resets the graph
		* @details This function allows to re-use the graph or update node potential, while preserving the graph structure.
		* It deletes all the nodes and edges and sets node index to zero.
		*/
		DllExport virtual void		reset(void) = 0;
		/**
		* @brief Adds an additional node (with specified potentional)
		* @param pot node potential vector: Mat(size: nStates x 1; type: CV_32FC1)
		* @return the node's ID
		*/
		DllExport virtual size_t	addNode(const Mat &pot = EmptyMat) = 0;
		/**
        * @brief Adds the graph nodes with potentials
        * @param pots A block of potentials: Mat(size: nNodes x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		addNodes(const Mat &pots);
		/**
		* @brief Sets or changes the potential of node
		* @param node node index
		* @param pot node potential vector: Mat(size: nStates x 1; type: CV_32FC1)
		*/
		DllExport virtual void		setNode(size_t node, const Mat &pot) = 0;
		/**
        * @brief Fills the graph nodes with new potentials
        * @details
        * > This function supports PPL
		* @param start_node The index of the node, starting from which the potentials should be set
		* @param pots A block of potentials: Mat(size: nNodes x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		setNodes(size_t start_node, const Mat &pots);
		/**
		* @brief Returns the node potential
		* @param[in] node node index
		* @param[out] pot node potential vector: Mat(size: nStates x 1; type: CV_32FC1)
		*/
		DllExport virtual void		getNode(size_t node, Mat &pot) const = 0;
		/**
		* @brief Returns the node potentials
        * @details
        * > This function supports PPL
		* @param[in] start_node The index of the node, starting from which the potentials should be got
		* @param[in] num_nodes The number of nodes potentials to acquire. \b 0 means - read nodes from \b start_node till the last one.
		* @param[out] pots A block of potentials: Mat(size: num_nodes x nStates; type: CV_32FC1)
		*/
		DllExport virtual void		getNodes(size_t start_node, size_t num_nodes, Mat &pots) const;
		/**
		* @brief Returns the set of IDs of the child nodes of the argument node
		* @param[in] node node index
		* @param[out] vNodes vector with the child node's ID
		*/
		DllExport virtual void      getChildNodes(size_t node, vec_size_t &vNodes) const = 0;
		/**
		* @brief Returns the set of IDs of the parent nodes of the argument node
		* @param[in] node node index
		* @param[out] vNodes vector with the parent node's ID
		*/
		DllExport virtual void      getParentNodes(size_t node, vec_size_t &vNodes) const = 0;		
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
		/**
		* @brief Returns number of states (classes)
		* @return Number of states (features)
		*/
		DllExport byte				getNumStates(void) const { return m_nStates; }

	
	private:
		byte m_nStates;		///< The number of states (classes)
	};
}
