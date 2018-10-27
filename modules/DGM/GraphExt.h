// Extended Graph abstract class interface;
// Written by Sergey Kosov in 2018 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
    // ================================ Extended Graph Class ================================
    /**
    * @brief Extended graph abstract class
    * @ingroup moduleGraph
    * @details This graph class provides additional functionality, when the graph is used for 2d image classification
    * @author Sergey G. Kosov, sergey.kosov@project-10.de
    */
    class CGraphExt
    {
    public:
        DllExport CGraphExt() = default;
        DllExport virtual ~CGraphExt() = default;

        /**
        * @brief Builds a 2d graph of size corresponding to the image resolution
        * @param graphSize The size of the graph
        */
        DllExport virtual void addNodes(Size graphSize) = 0;
        /**
        * @brief Fills the graph nodes with potentials
        * @param pots A block of node potentials: Mat(type: CV_32FC(nStates)). It may be obtained by:
        * @code
        * CTrainNode::getNodePotentials()
        * @endcode
        */
        DllExport virtual void setNodes(const Mat& pots) = 0;
        /**
		* @brief Adds default data-independet edge model
		*/
		DllExport virtual void addDefaultEdgesModel() = 0;
		/**
		* @brief Adds default contrast-sensitive edge model
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
		*/
		DllExport virtual void addDefaultEdgesModel(const Mat &featureVectors) = 0;
		/**
        * @brief Returns the size of the graph
        * @return The size of the Graph
        */
        DllExport virtual Size getSize() const = 0;


    private:
        // Copy semantics are disabled
        CGraphExt(const CGraphExt&) = delete;
        const CGraphExt& operator=(const CGraphExt& rhs) = delete;

    };
}