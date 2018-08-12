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
	class IGraph
	{
	public:
		/**
		* @brief Constructor
		* @param nStates the number of States (classes)
		*/
		DllExport IGraph(byte nStates) : m_nStates(nStates) {};
		DllExport virtual ~IGraph(void) {};

		/**
		* @brief Resets the graph
		* @details This function allows to re-use the graph or update node potential, while preserving the graph structure.
		* It deletes all the nodes and edges and sets node index to zero.
		*/
		DllExport virtual void		reset(void) = 0;
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
		DllExport size_t			getNumStates(void) const { return m_nStates; }

	protected:
		byte m_nStates;		///< The number of states (classes)


	private:
		// Copy semantics are disabled
		IGraph(const IGraph &rhs) {}
		const IGraph & operator= (const IGraph & rhs) { return *this; }
	};
}