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
    * @details This class provides wrapper functions for simplifying operations with graphs, when they are used for 2d image classification
    * @author Sergey G. Kosov, sergey.kosov@project-10.de
    */
    class CGraphExt
    {
    public:
        DllExport CGraphExt(void) = default;
        DllExport CGraphExt(const CGraphExt&) = delete;
		DllExport virtual ~CGraphExt(void) = default;
        DllExport const CGraphExt& operator=(const CGraphExt&) = delete;

        /**
        * @brief Builds a 2D graph of size corresponding to the image resolution
        * @param graphSize The size of the graph
        */
        DllExport virtual void addNodes(Size graphSize) = 0;
        /**
        * @brief Fills the existing graph nodes with potentials or adds new nodes with potentials
        * @details
        * If the graph was not build beforehand, this function calls first
        * @code
        * addNodes(pots.size())
        * @endcode
        * @param pots A block of node potentials: Mat(type: CV_32FC(nStates)). It may be obtained by:
        * @code
        * CTrainNode::getNodePotentials()
        * @endcode
        */
        DllExport virtual void setNodes(const Mat& pots) = 0;
        /**
		* @brief Adds default data-independet edge model
		* @param val Value, specifying the smoothness strength 
		* > See concrete class implementation for more details
        * @param weight The weighting parameter
		*/
		DllExport virtual void addDefaultEdgesModel(float val, float weight = 1.0f) = 0;
		/**
		* @brief Adds default contrast-sensitive edge model
		* @param featureVectors Multi-channel matrix, each element of which is a multi-dimensinal point: Mat(type: CV_8UC<nFeatures>)
        * @param val Value, specifying the smoothness strength
		* > See concrete class implementation for more details
        * @param weight The weighting parameter
		*/
		DllExport virtual void addDefaultEdgesModel(const Mat &featureVectors, float val, float weight = 1.0f) = 0;
		/**
        * @brief Adds default contrast-sensitive edge model
        * @param featureVectors Vector of size \a nFeatures, each element of which is a single feature - image: Mat(type: CV_8UC1)
        * @param val Value, specifying the smoothness strength
		* > See concrete class implementation for more details
        * @param weight The weighting parameter
        */
        DllExport virtual void addDefaultEdgesModel(const vec_mat_t &featureVectors, float val, float weight = 1.0f) = 0;
        /**
        * @brief Returns the size of the graph
        * @return The size of the Graph
        */
        DllExport virtual Size getSize() const = 0;
    };
}