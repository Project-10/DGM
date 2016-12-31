// Nested training model for pairwise link potentials
// Written by Sergey G. Kosov in 2016 for Project X
#pragma once

#include "TrainLink.h"

namespace DirectGraphicalModels
{
	class CPriorNode;
	class CTrainNode;
	
	// ============================= Nested Link Train Class =============================
	/**
	* @ingroup moduleTrainEdge
	* @brief Nested link (inter-layer edge) training class
	* @details This class in order to estimate edge potentials, uses an unary potential trainer, which is learned for \b nStatesBase \f$\times\f$ \b nStatesOccl states (classes),
	* corresponding to the base and occlusion layers of the graphical model.
	* @tparam Trainer The nested node potential trainer, derived from the CTrainNode class
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	template<class Trainer> class CTrainLinkNested : public CTrainLink
	{
	public:
		/**
		* @brief Constructor
		* @param nStatesBase Number of states (classes) for the base layer of the graphical model
		* @param nStatesOccl Number of states (classes) for the occlusion layer of the graphical model
		* > \b nStatesBase * \b nStatesOccl must be less then 256.
		* @param nFeatures Number of features
		*/
		CTrainLinkNested(byte nStatesBase, byte nStatesOccl, word nFeatures) 
			: CTrainLink(nStatesBase, nStatesOccl, nFeatures)
			, CBaseRandomModel(nStatesBase * nStatesOccl)
		{
			word nStates = static_cast<word>(nStatesBase) * static_cast<word>(nStatesOccl);
			DGM_ASSERT(nStates < 256);
			m_pPrior	= new CPriorNode(static_cast<byte>(nStates));
			m_pTrainer	= new Trainer(static_cast<byte>(nStates), nFeatures);
			
		}
		/**
		* @brief Constructor
		* @tparam TrainerParams Type of the parameters to the nested node potential trainer
		* @param nStatesBase Number of states (classes) for the base layer of the graphical model
		* @param nStatesOccl Number of states (classes) for the occlusion layer of the graphical model
		* @param nFeatures Number of features
		* @param params Parameters of the nested node potential trainer
		*/
		template<class TrainerParams> CTrainLinkNested(byte nStatesBase, byte nStatesOccl, word nFeatures, TrainerParams params)
			: CTrainLink(nStatesBase, nStatesOccl, nFeatures)
			, CBaseRandomModel(nStatesBase * nStatesOccl)
		{
			word nStates = static_cast<word>(nStatesBase) * static_cast<word>(nStatesOccl);
			DGM_ASSERT(nStates < 256);
			m_pPrior = new CPriorNode(static_cast<byte>(nStates));
			m_pTrainer = new Trainer(static_cast<byte>(nStates), nFeatures, params);
		}
		
		virtual ~CTrainLinkNested(void) 
		{ 
			delete m_pPrior;
			delete m_pTrainer; 
		}

		virtual void	reset(void) { m_pPrior->reset(); m_pTrainer->reset(); }
		virtual void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const { m_pTrainer->save(path, name.empty() ? "CTrainLinkNested" : name, idx); }
		virtual void	load(const std::string &path, const std::string &name = std::string(), short idx = -1) { m_pTrainer->load(path, name.empty() ? "CTrainLinkNested" : name, idx); }

		virtual void	addFeatureVec(const Mat &featureVector, byte gtb, byte gto)
		{
			byte gt = gtb + m_nStatesBase * gto;
			m_pPrior->addNodeGroundTruth(gt);
			m_pTrainer->addFeatureVec(featureVector, gt);
		}
		
		virtual void	train(bool doClean = false)
		{ 
			// Fill holes in trainig
			Mat fv(m_nFeatures, 1, CV_8UC1, Scalar(0));
			Mat priors = m_pPrior->getPrior();
			for (byte i = 0; i < priors.rows; i++)
				if (priors.at<float>(i, 0) == 0)
					m_pTrainer->addFeatureVec(fv, i);
				
			
			m_pTrainer->train(doClean); 
		}


	protected:
		virtual void	saveFile(FILE *pFile) const {}
		virtual void	loadFile(FILE *pFile) {}
		/**
		* @brief Returns the data-dependent link (inter-layer edge) potentials
		* @details This function returns edge potential matrix, which elements are obrained from the unary potential vector:
		* \f[ edgePot[nStatesBase:nStates][0:nStatesBase] = nodePot[nStatesBase \times nStatesOccl] = f(\textbf{f}[nFeatures]). \f]
		* Here \f$ nStates = nStatesBase + nStatesOccl\f$.
		* @param featureVector Multi-dimensinal point \f$\textbf{f}\f$: Mat(size: nFeatures x 1; type: CV_8UC1), corresponding to the data site of the both edge nodes
		* @return The edge potential matrix: Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		virtual Mat	calculateLinkPotentials(const Mat &featureVector) const
		{
			Mat pot = m_pTrainer->getNodePotentials(featureVector);
			//pot = m_pPrior->getPrior(100);

			DGM_ASSERT_MSG(pot.rows == m_nStatesBase * m_nStatesOccl, "The length of the node potentinal vector = %d, but must be %d", pot.rows, m_nStatesBase * m_nStatesOccl);

			Mat res(m_nStatesBase + m_nStatesOccl, m_nStatesBase + m_nStatesOccl, CV_32FC1, Scalar(0));

			for (register byte gto = 0; gto < m_nStatesOccl; gto++) {
				float * pRes = res.ptr<float>(m_nStatesBase + gto);
				for (register byte gtb = 0; gtb < m_nStatesBase; gtb++) {
					byte gt = gtb + m_nStatesBase * gto;
					pRes[gtb] = pot.at<float>(gt, 0);
				}
			}

			return res;
		}


	protected:
		CPriorNode				* m_pPrior;				///< %Node prior poobability
		CTrainNode				* m_pTrainer;			///< %Node trainer

	};
}