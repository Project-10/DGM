// Abstract Kit classes for constructing the Graph instances
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels 
{
	/// Types of the graphical model
	enum class GraphType { 
		pairwise,		///< Pairwise graph
		dense			///< Dense (complete) graph
	};
	
	class CGraph;
	class CInfer;
	class CGraphExt;

	// ================================ Graph Kit Abstract Factory Class ===============================
	/**
	* @brief Abstract Kit class for constructing Graph-related objects
	* @ingroup moduleGraphKit
	* @details The derived classes are aught to create compatible objects for graph building and inference / decoding. This abstract class provides a standard interface for accessing 
	* these objects.
	* @author Dr. Sergey Kosov, sergey.kosov@project-10.de
	*/
	class CGraphKit {
	public:
		DllExport CGraphKit() = default;
		DllExport CGraphKit(const CGraphKit&) = delete;
		DllExport virtual ~CGraphKit() = default;
		DllExport const CGraphKit& operator=(const CGraphKit&) = delete;

		/**
		* @brief Factory method returning graph kit object
		* @note The resulting graph kit object is created with default parameters
		* @param graphType Type of the desired graphical model (Ref. @ref GraphType)
		* @param nStates The number of States (classes)
		* @return Tne pointer to the concrete implementation of the graph kit class
		*/
		DllExport static std::shared_ptr<CGraphKit> create(GraphType graphType, byte nStates);
		/**
		* @brief Returns the graph object 
		* @return The reference to the graph object
		*/
		DllExport virtual CGraph&		getGraph() = 0;
		/**
		* @brief Returns the inference / decoding object
		* @return The reference to the inference / decoding object
		*/
		DllExport virtual CInfer&		getInfer() = 0;
		/**
		* @brief Returns the graph extension object
		* @return The reference to the graph extension object
		*/
		DllExport virtual CGraphExt&	getGraphExt() = 0;
	};
}