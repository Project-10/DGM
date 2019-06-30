// Potts Edge Model class interface
// Written by Sergey G. Kosov in 2019 for Project X 
#pragma once

#include "IEdgeModel.h"

class CPermutohedral;

namespace DirectGraphicalModels {
	// ================================ Potts Edge Model ================================
	/**
	* @brief Potts %Edge Model for dense graphical models
	* @details This class implements Potts edge potential model in the fully connected CRF. The implementation is based
	* on <a href="http://graphics.stanford.edu/projects/densecrf/densecrf.pdf">Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials</a> paper.
	* @ingroup moduleGraph
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CEdgeModelPotts : public IEdgeModel {
	public:
		/**
		* @brief Constructor
		* @details This constucts a new edge potentials model, which is in general \a training-data-independent, but still 
		* \a contrast-sensitive model (see PhD Thesis <a href="http://www.project-10.de/Kosov/files/doctoralthesis.pdf">Multi-Layer Conditional Random Fields for Revealing Unobserved Entities</a>
		* p. 60 for more details). \a Test-data-dependency is provided via \b features argument. However this model might be ''trained'' by using a semimetric function provided via \b semiMetricFunction
		* argument.
		* @param features The set of features which correspond to the nodes of the dense graphical model: Mat(size: nNodes x nFeatures; type: CV_32FC1)
		* @param weight The weighting parameter (default value is 1)
		* @param semiMetricFunction Reference to a semi-metric function, which arguments \b src and \b dst are: Mat(size: 1 x nFeatures; type: CV_32FC1). This function when provided 
		* will be called for every node potential in the apply() method. 
		* @param perPixelNormalization Flag indicating whether er-pixel normalization should be used during applying the edge model.
		*/
		DllExport CEdgeModelPotts(const Mat& features, float weight = 1.0f, const std::function<void(const Mat& src, Mat& dst)>& semiMetricFunction = {}, bool perPixelNormalization = true);
		DllExport virtual ~CEdgeModelPotts(void);
	
		DllExport void apply(const Mat &src, Mat &dst) const override;
	

	private:
		CPermutohedral								  * m_pLattice;		///< Pointer to the permutohedral lattice
		float											m_weight;		///< The weighting parameter
		Mat												m_norm;			///< Array with normalization factors
		std::function<void(const Mat &src, Mat &dst)>	m_function;		///< The semi-metric function
	};
}