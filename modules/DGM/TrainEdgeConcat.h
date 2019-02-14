// Concatenated training model for pairwise potentials
// Written by Sergey G. Kosov in 2015 - 2016 for Project X
#pragma once

#include "TrainEdge.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	class CPriorNode;
	class CTrainNode;
	class CFeaturesConcatenator;
	
	// ============================= Concatenated Edge Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Concatenated edge training class
	* @details This class in order to estimate edge potentials, uses an unary potential trainer, which is learned for \f$nStates^2\f$ states (classes).
	* Both edge feature vectors are concatenated in one feature vector, which in its turn, is used for the unary potential trainer. The feature concatenation
	* is performed by the @ref CFeaturesConcatenator class.
	* @tparam Trainer The nested node potential trainer, derived from the CTrainNode class
	* @tparam Concatenator The feature concatenator, derived from the CFeaturesConcatenator class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	template<class Trainer, class Concatenator> class CTrainEdgeConcat : public CTrainEdge
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		* > Must be less then 16
		* @param nFeatures Number of features
		*/
		CTrainEdgeConcat(byte nStates, word nFeatures) 
			: CBaseRandomModel(nStates)
            , CTrainEdge(nStates, nFeatures)
		{
			DGM_ASSERT(nStates < 16);
			m_pPrior		= new CPriorNode(nStates * nStates);
			m_pTrainer		= new Trainer(nStates * nStates, nFeatures);
			m_pConcatenator = new Concatenator(nFeatures);
			m_featureVector = Mat(m_pConcatenator->getNumFeatures(), 1, CV_8UC1);
		}
		/**
		* @brief Constructor
		* @tparam TrainerParams Type of the parameters to the nested node potential trainer
		* @param nStates Number of states (classes)
		* > Must be less then 16
		* @param nFeatures Number of features
		* @param params Parameters of the nested node potential trainer
		*/
		template<class TrainerParams> CTrainEdgeConcat(byte nStates, word nFeatures, TrainerParams params)
			: CTrainEdge(nStates, nFeatures)
			, CBaseRandomModel(nStates)
		{
			DGM_ASSERT(nStates < 16);
			m_pPrior		= new CPriorNode(nStates * nStates);
			m_pTrainer		= new Trainer(nStates * nStates, nFeatures, params);
			m_pConcatenator = new Concatenator(nFeatures);
			m_featureVector = Mat(m_pConcatenator->getNumFeatures(), 1, CV_8UC1);
		}


		virtual ~CTrainEdgeConcat(void) 
		{
			delete m_pPrior;
			delete m_pTrainer;
			delete m_pConcatenator;
		}

		virtual void	reset(void) { m_pPrior->reset(); m_pTrainer->reset(); }
		void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const { m_pTrainer->save(path, name.empty() ? "CTrainEdgeConcat" : name, idx); }
		void	load(const std::string &path, const std::string &name = std::string(), short idx = -1) { m_pTrainer->load(path, name.empty() ? "CTrainEdgeConcat" : name, idx); }

		virtual void	addFeatureVecs(const Mat &featureVector1, byte gt1, const Mat &featureVector2, byte gt2) 
		{
			byte gt = gt2 * m_nStates + gt1;
			m_pPrior->addNodeGroundTruth(gt);
			m_pConcatenator->concatenate(featureVector1, featureVector2, m_featureVector);
			m_pTrainer->addFeatureVec(m_featureVector, gt);
		}
		virtual void	train(bool doClean = false) { m_pTrainer->train(doClean); }


	protected:
		DllExport virtual void	saveFile(FILE *pFile) const {} 
		DllExport virtual void	loadFile(FILE *pFile) {} 		
		/**
		* @brief Returns the data-dependent edge potentials
		* @details This function returns edge potential matrix, which elements are obrained from the unary potential vector: 
		* \f[ edgePot[nStates][nStates] = nodePot[nStates^2] = f(concat(\textbf{f}_1[nFeatures],\textbf{f}_2[nFeatures])). \f]
		* The resulting edge potential matrix is normalized such that its largest element is equal to paramter \f$\theta\f$, and then regularized as follows:
		* \f[ edgePot_{s,s} = \left\{\begin{array}{ll} 1 &\mbox{ if $edgePot_{s,s}<1$} \\ edgePot_{s,s} &\mbox{ otherwise}\end{array} \right.\;\;\;\forall s\in\mathbb{S}.\f]
		* @todo: Incorporate the node potential weight into the model parameters
		* @param featureVector1 Multi-dimensinal point \f$\textbf{f}_1\f$: Mat(size: nFeatures x 1; type: CV_8UC1), corresponding to the first node of the edge
		* @param featureVector2 Multi-dimensinal point \f$\textbf{f}_2\f$: Mat(size: nFeatures x 1; type: CV_8UC1), corresponding to the second node of the edge
		* @param vParams Array of control parameters \f$\vec{\theta}\f$, which must consist from \a one parameter, specifying the largest value in the resulting edge potential.
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport virtual Mat	calculateEdgePotentials(const Mat &featureVector1, const Mat &featureVector2, const vec_float_t &vParams) const 
		{
			const float nodePotWeight = 1.0f;
			m_pConcatenator->concatenate(featureVector1, featureVector2, const_cast<Mat &>(m_featureVector));
			Mat pot = m_pTrainer->getNodePotentials(m_featureVector, nodePotWeight);
			Mat prior = m_pPrior->getPrior(100);

			Mat res(m_nStates, m_nStates, CV_32FC1);
			
			for (byte gt1 = 0; gt1 < m_nStates; gt1++) {
				float * pRes = res.ptr<float>(gt1);
				for (byte gt2 = 0; gt2 < m_nStates; gt2++) {
					byte gt = gt2 * m_nStates + gt1;
					
					float epsilon = prior.at<float>(gt) > 0 ? FLT_EPSILON : 0.0f;
					pRes[gt2] = MAX(pot.at<float>(gt), epsilon);
				}
			}

			return res;
		}
	

	private:
		CPriorNode				* m_pPrior;				///< %Node prior poobability
		CTrainNode				* m_pTrainer;			///< %Node trainer
		CFeaturesConcatenator	* m_pConcatenator;		///< Feature concatenator
		Mat						  m_featureVector;		///< Feature vector
	};
}
