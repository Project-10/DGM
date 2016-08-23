#include "GraphLayered.h"
#include "TrainNode.h"
#include "TrainEdge.h"
#include "TrainLink.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CGraphLayered::build(CvSize graphSize)
	{
		if (getNumNodes() != 0) reset();
		
		size_t l;
		for (int y = 0; y < graphSize.height; y++)
			for (int x = 0; x < graphSize.width; x++) {
				// Nodes
				size_t idx = addNode();
				for (l = 1; l < m_nLayers; l++) addNode();

				if (m_gType & GRAPH_EDGES_LINK) {
					word nLayers = MIN(2, m_nLayers);
					for (l = 0; l < nLayers - 1; l++)
						addArc(idx + l, idx + l + 1);
					//for (l = nLayers - 1; l < m_nLayers - 1; l++)
					//	addEdge(idx + l, idx + l + 1);
				}

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0)
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers);

					if (y > 0)
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers * graphSize.width);
				}

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers * graphSize.width - m_nLayers);

					if ((x < graphSize.width - 1) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers * graphSize.width + m_nLayers);
				}
			} // x
	}

	void CGraphLayered::fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const Mat &featureVectors, float weightBase, float weightOccl)
	{
		const int	height		= featureVectors.rows;
		const int	width		= featureVectors.cols;
		const word	nFeatures	= featureVectors.channels();

		// Assertions
		DGM_ASSERT(nFeatures == nodeTrainerBase->getNumFeatures());
		if (nodeTrainerOccl) DGM_ASSERT(nFeatures == nodeTrainerOccl->getNumFeatures());
		DGM_ASSERT(width * height * m_nLayers == getNumNodes());

		Rect ROIb(0, 0, 1, nodeTrainerBase->getNumStates());
		Rect ROIo = nodeTrainerOccl ? cvRect(0, m_nStates - nodeTrainerOccl->getNumStates(), 1, nodeTrainerOccl->getNumStates()) : cvRect(0, 0, 0, 0);

#ifdef USE_PPL
		concurrency::parallel_for(0, height, [&, width, nFeatures, ROIb, ROIo](int y) {
			Mat featureVector(nFeatures, 1, CV_8UC1);
			Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
#else
		Mat featureVector(nFeatures, 1, CV_8UC1);
		Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		for (int y = 0; y < height; y++) {
#endif
			const byte *pFv = featureVectors.ptr<byte>(y);
			int i = y * width * m_nLayers;
			for (int x = 0; x < width; x++) {
				for (word f = 0; f < nFeatures; f++) featureVector.at<byte>(f, 0) = pFv[nFeatures * x + f];
				nodeTrainerBase->getNodePotentials(featureVector, weightBase).copyTo(nPotBase(ROIb));
				setNode(i, nPotBase);
				if (m_nLayers >= 2) {
					nodeTrainerOccl->getNodePotentials(featureVector, weightOccl).copyTo(nPotOccl(ROIo));
					setNode(i + 1, nPotOccl);
				}
				i += m_nLayers;
			} // x	
#ifdef USE_PPL
		}); // y
#else
		} // y
#endif
	}

	void CGraphLayered::fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const vec_mat_t &featureVectors, float weightBase, float weightOccl)
	{
		const int	height		= featureVectors[0].rows;
		const int	width		= featureVectors[0].cols;
		const word	nFeatures	= static_cast<word>(featureVectors.size());

		// Assertions
		DGM_ASSERT(nFeatures == nodeTrainerBase->getNumFeatures());
		if (nodeTrainerOccl) DGM_ASSERT(nFeatures == nodeTrainerOccl->getNumFeatures());
		DGM_ASSERT(width * height * m_nLayers == getNumNodes());

		Rect ROIb(0, 0, 1, nodeTrainerBase->getNumStates());
		Rect ROIo = nodeTrainerOccl ? cvRect(0, m_nStates - nodeTrainerOccl->getNumStates(), 1, nodeTrainerOccl->getNumStates()) : cvRect(0, 0, 0, 0);

#ifdef USE_PPL
		concurrency::parallel_for(0, height, [&, width, nFeatures, ROIb, ROIo](int y) {
			Mat featureVector(nFeatures, 1, CV_8UC1);
			Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
#else
		Mat featureVector(nFeatures, 1, CV_8UC1);
		Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		for (int y = 0; y < height; y++) {
#endif
			byte const **pFv = new const byte *[nFeatures];
			for (word f = 0; f < nFeatures; f++) pFv[f] = featureVectors[f].ptr<byte>(y);
			int i = y * width * m_nLayers;
			for (int x = 0; x < width; x++) {
				for (word f = 0; f < nFeatures; f++) featureVector.at<byte>(f, 0) = pFv[f][x];
				Mat nnnPot = nodeTrainerBase->getNodePotentials(featureVector, weightBase);
				nnnPot.copyTo(nPotBase(ROIb));
				setNode(i, nPotBase);
				if (m_nLayers >= 2) { 
					nodeTrainerOccl->getNodePotentials(featureVector, weightOccl).copyTo(nPotOccl(ROIo));
					setNode(i + 1, nPotOccl);
				}
				i += m_nLayers;
			} // x	
#ifdef USE_PPL
		}); // y
#else
		} // y
#endif	
	}

	void CGraphLayered::fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const Mat &featureVectors, float *params, size_t params_len, float edgeWeight, float linkWeight)
	{
		const int	height		= featureVectors.rows;
		const int	width		= featureVectors.cols;
		const word	nFeatures	= featureVectors.channels();

		// Assertions
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(width * height * m_nLayers == getNumNodes());

#ifdef USE_PPL
		concurrency::parallel_for(0, height, [&, width, nFeatures](int y) {
			Mat featureVector1(nFeatures, 1, CV_8UC1);
			Mat featureVector2(nFeatures, 1, CV_8UC1);
			Mat ePot;
			word l;
#else 
		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);
		Mat ePot;
		word l;
		for (int y = 0; y < height; y++) {
#endif
			const byte *pFv1 = featureVectors.ptr<byte>(y);
			const byte *pFv2 = (y > 0) ? featureVectors.ptr<byte>(y - 1) : NULL;
			int i = y * width * m_nLayers;
			for (int x = 0; x < width; x++) {
				for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];				// featureVectors[x][y]
				
				if (m_gType & GRAPH_EDGES_LINK) {
					ePot = linkTrainer->getLinkPotentials(featureVector1, linkWeight);
					add(ePot, ePot.t(), ePot);
					word nLayers = MIN(2, m_nLayers);
					for (l = 0; l < nLayers - 1; l++)
						setArc(i + l, i + l + 1, ePot);
					//for (l = nLayers - 1; l < m_nLayers - 1; l++)
					//	setEdge(i + l, i + l + 1);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers * width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x - 1) + f];	// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers * width - m_nLayers, ePot);
					} // if x, y

					if ((x < width - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x + 1) + f];	// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers * width + m_nLayers, ePot);
					} // x, y
				} // edges_diag
				i += m_nLayers;
			} // x
#ifdef USE_PPL
		}); // y
#else
		} // y
#endif
	}
	
	void CGraphLayered::fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const vec_mat_t &featureVectors, float *params, size_t params_len, float edgeWeight, float linkWeight)
	{
		const int	height		= featureVectors[0].rows;
		const int	width		= featureVectors[0].cols;
		const word	nFeatures	=static_cast<word>(featureVectors.size());

		// Assertions
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(width * height * m_nLayers == getNumNodes());

#ifdef USE_PPL
		concurrency::parallel_for(0, height, [&, width, nFeatures](int y) {
			Mat featureVector1(nFeatures, 1, CV_8UC1);
			Mat featureVector2(nFeatures, 1, CV_8UC1);
			Mat ePot;
			word l;
#else 
		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);
		Mat ePot;
		word l;
		for (int y = 0; y < height; y++) {
#endif
			byte const **pFv1 = new const byte * [nFeatures];
			for (word f = 0; f < nFeatures; f++) pFv1[f] = featureVectors[f].ptr<byte>(y);
			byte const **pFv2 = NULL;
			if (y > 0) {
				pFv2 = new const byte *[nFeatures];
				for (word f = 0; f < nFeatures; f++) pFv2[f] = featureVectors[f].ptr<byte>(y-1);
			}
			
			int i = y * width * m_nLayers;
			for (int x = 0; x < width; x++) {
				for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[f][x];				// featureVectors[x][y]

				if (m_gType & GRAPH_EDGES_LINK) {
					ePot = linkTrainer->getLinkPotentials(featureVector1, linkWeight);
					add(ePot, ePot.t(), ePot);
					word nLayers = MIN(2, m_nLayers);
					for (l = 0; l < nLayers - 1; l++)
						setArc(i + l, i + l + 1, ePot);
					//for (l = nLayers - 1; l < m_nLayers - 1; l++)
					//	setEdge(i + l, i + l + 1);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[f][x - 1];				// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x];					// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers * width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x - 1];				// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers * width - m_nLayers, ePot);
					} // if x, y

					if ((x < width - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x + 1];				// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(i + l, i + l - m_nLayers * width + m_nLayers, ePot);
					} // x, y
				} // edges_diag
				i += m_nLayers;
			} // x
#ifdef USE_PPL
		}); // y
#else
		} // y
#endif
	}

	void CGraphLayered::marginalize(const vec_size_t &nodes)
	{
		Mat pot, pot1, pot2;

		for (size_t node : nodes) {
			vec_size_t parentNodes, childNodes, managers;
			getParentNodes(node, parentNodes);
			getChildNodes(node, childNodes);
			
			// find all managers for the node
			for (size_t child : childNodes) {
				// Looking for those child nodes, which are managers
				auto isArc = std::find(parentNodes.begin(), parentNodes.end(), child);	// If there is a return edge => the child is a neighbor
				if (isArc != parentNodes.end()) continue;								
				// Here the child is a manager
				auto isInZ = std::find(nodes.begin(), nodes.end(), child);				// If the manager is to be also marginalized 
				if (isInZ != nodes.end()) continue;

				managers.push_back(child);

				// Add new edges (from any other neighboring node to the manager)
				for (size_t parent : parentNodes) {
					auto isInZ = std::find(nodes.begin(), nodes.end(), parent);			// If the parent is to be also marginalized 
					if (isInZ != nodes.end()) continue;
					
					getEdge(parent, node, pot1);
					getEdge(node, child, pot2);
					if (pot1.empty() && pot2.empty()) addEdge(parent, child);
					else {
						pot1.empty() ? pot = pot2 + pot1 : pot = pot1 + pot2;
						addEdge(parent, child, pot);
					}
				}
			}

			// Add new arcs (between two managers)
			if (managers.size() >= 2)
				for (size_t i = 0; i < managers.size() - 1; i++)
					for (size_t j = i + 1; j < managers.size(); j++) {
						getEdge(node, managers[i], pot1);
						getEdge(node, managers[j], pot2);
						if (pot1.empty() && pot2.empty())	addArc(managers[i], managers[j]);
						else {
							pot1.empty() ? pot = pot2 + pot1 : pot = pot1 + pot2;
							addArc(managers[i], managers[j], pot);
						}
					}


			// Delete all	
			for (size_t &parent : parentNodes) removeEdge(parent, node);
			for (size_t &child : childNodes)  removeEdge(node, child);
		} // n

	}

}