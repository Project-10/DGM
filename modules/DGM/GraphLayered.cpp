#include "GraphLayered.h"
#include "TrainNode.h"
#include "TrainEdge.h"
#include "TrainEdgePotts.h"
#include "TrainLink.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CGraphLayered::build(CvSize graphSize)
	{
		if (getNumNodes() != 0) reset();
		m_size = graphSize;

		word l;
		for (int y = 0; y < m_size.height; y++)
			for (int x = 0; x < m_size.width; x++) {
				// Nodes
				size_t idx = addNode();
				for (l = 1; l < m_nLayers; l++) addNode();

				if (m_gType & GRAPH_EDGES_LINK) {
					if (m_nLayers >= 2)
						addArc(idx, idx + 1);
					for (l = 2; l < m_nLayers; l++)
						addEdge(idx + l - 1, idx + l);
				} // if LINK

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0)
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers);
					if (y > 0)
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers * m_size.width);
				} // if GRID
			} // x


		if (m_gType & GRAPH_EDGES_DIAG) {
			for (int y = 0; y < m_size.height; y++) {
				for (int x = 0; x < m_size.width; x++) {
					size_t idx = (y * m_size.width + x) * m_nLayers;

					if ((x > 0) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers * (m_size.width + 1));

					if ((x < graphSize.width - 1) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							addArc(idx + l, idx + l - m_nLayers * (m_size.width - 1));
				} // x
			} // y
		} // if DIAG
	}

	void CGraphLayered::fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const Mat &featureVectors, float weightBase, float weightOccl)
	{
		const word	nFeatures	= featureVectors.channels();

		// Assertions
		DGM_ASSERT(m_size.height == featureVectors.rows);
		DGM_ASSERT(m_size.width == featureVectors.cols);
		DGM_ASSERT(nFeatures == nodeTrainerBase->getNumFeatures());
		if (nodeTrainerOccl) DGM_ASSERT(nFeatures == nodeTrainerOccl->getNumFeatures());
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == getNumNodes());

		Rect ROIb(0, 0, 1, nodeTrainerBase->getNumStates());
		Rect ROIo = nodeTrainerOccl ? cvRect(0, m_nStates - nodeTrainerOccl->getNumStates(), 1, nodeTrainerOccl->getNumStates()) : cvRect(0, 0, 0, 0);

#ifdef ENABLE_PPL
		concurrency::parallel_for(0, m_size.height, [&, nFeatures, ROIb, ROIo] (int y) {
			Mat featureVector(nFeatures, 1, CV_8UC1);
			Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotIntr(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			if (nodeTrainerOccl) nPotIntr(ROIo).setTo(100.0f / nodeTrainerOccl->getNumStates());
#else
		Mat featureVector(nFeatures, 1, CV_8UC1);
		Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotIntr(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		if (nodeTrainerOccl) nPotIntr(ROIo).setTo(100.0f / nodeTrainerOccl->getNumStates());
		for (int y = 0; y < m_size.height; y++) {
#endif
			const byte *pFv = featureVectors.ptr<byte>(y);
			for (int x = 0; x < m_size.width; x++) {
				size_t idx = (y * m_size.width + x) * m_nLayers;
				for (word f = 0; f < nFeatures; f++) featureVector.at<byte>(f, 0) = pFv[nFeatures * x + f];
				nodeTrainerBase->getNodePotentials(featureVector, weightBase).copyTo(nPotBase(ROIb));
				setNode(idx, nPotBase);
				if (m_nLayers >= 2) {
					nodeTrainerOccl->getNodePotentials(featureVector, weightOccl).copyTo(nPotOccl(ROIo));
					setNode(idx + 1, nPotOccl);
				}
				for (word l = 2; l < m_nLayers; l++)
					setNode(idx + l, nPotIntr);
			} // x	
#ifdef ENABLE_PPL
		}); // y
#else
		} // y
#endif
	}

	void CGraphLayered::fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const vec_mat_t &featureVectors, float weightBase, float weightOccl)
	{
		const word	nFeatures	= static_cast<word>(featureVectors.size());

		// Assertions
		DGM_ASSERT(m_size.height == featureVectors[0].rows);
		DGM_ASSERT(m_size.width == featureVectors[0].cols);
		DGM_ASSERT(nFeatures == nodeTrainerBase->getNumFeatures());
		if (nodeTrainerOccl) DGM_ASSERT(nFeatures == nodeTrainerOccl->getNumFeatures());
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == getNumNodes());

		Rect ROIb(0, 0, 1, nodeTrainerBase->getNumStates());
		Rect ROIo = nodeTrainerOccl ? cvRect(0, m_nStates - nodeTrainerOccl->getNumStates(), 1, nodeTrainerOccl->getNumStates()) : cvRect(0, 0, 0, 0);

#ifdef ENABLE_PPL
		concurrency::parallel_for(0, m_size.height, [&, nFeatures, ROIb, ROIo](int y) {
			Mat featureVector(nFeatures, 1, CV_8UC1);
			Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotIntr(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			if (nodeTrainerOccl) nPotIntr(ROIo).setTo(100.0f / nodeTrainerOccl->getNumStates());
#else
		Mat featureVector(nFeatures, 1, CV_8UC1);
		Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotIntr(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		if (nodeTrainerOccl) nPotIntr(ROIo).setTo(100.0f / nodeTrainerOccl->getNumStates());
		for (int y = 0; y < m_size.height; y++) {
#endif
			byte const **pFv = new const byte *[nFeatures];
			for (word f = 0; f < nFeatures; f++) pFv[f] = featureVectors[f].ptr<byte>(y);
			for (int x = 0; x < m_size.width; x++) {
				size_t idx = (y * m_size.width + x) * m_nLayers;
				for (word f = 0; f < nFeatures; f++) featureVector.at<byte>(f, 0) = pFv[f][x];
				nodeTrainerBase->getNodePotentials(featureVector, weightBase).copyTo(nPotBase(ROIb));
				setNode(idx, nPotBase);
				if (m_nLayers >= 2) { 
					nodeTrainerOccl->getNodePotentials(featureVector, weightOccl).copyTo(nPotOccl(ROIo));
					setNode(idx + 1, nPotOccl);
				}
				for (word l = 2; l < m_nLayers; l++)
					setNode(idx + l, nPotIntr);
			} // x	
#ifdef ENABLE_PPL
		}); // y
#else
		} // y
#endif	
	}

	void CGraphLayered::fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const Mat &featureVectors, float *params, size_t params_len, float edgeWeight, float linkWeight)
	{
		const word	nFeatures	= featureVectors.channels();

		// Assertions
		DGM_ASSERT(m_size.height == featureVectors.rows);
		DGM_ASSERT(m_size.width == featureVectors.cols);
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == getNumNodes());

#ifdef ENABLE_PPL
		concurrency::parallel_for(0, m_size.height, [&, nFeatures](int y) {
			Mat featureVector1(nFeatures, 1, CV_8UC1);
			Mat featureVector2(nFeatures, 1, CV_8UC1);
			Mat ePot;
			word l;
#else 
		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);
		Mat ePot;
		word l;
		for (int y = 0; y < m_size.height; y++) {
#endif
			const byte *pFv1 = featureVectors.ptr<byte>(y);
			const byte *pFv2 = (y > 0) ? featureVectors.ptr<byte>(y - 1) : NULL;
			for (int x = 0; x < m_size.width; x++) {
				size_t idx = (y * m_size.width + x) * m_nLayers;
				for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];				// featureVectors[x][y]
				
				if (m_gType & GRAPH_EDGES_LINK) {
					ePot = linkTrainer->getLinkPotentials(featureVector1, linkWeight);
					add(ePot, ePot.t(), ePot);
					if (m_nLayers >= 2)
						setArc(idx, idx + 1, ePot);
					ePot = CTrainEdgePotts::getEdgePotentials(100, m_nStates);
					for (l = 2; l < m_nLayers; l++)
						setEdge(idx + l - 1, idx + l, ePot);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers * m_size.width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x - 1) + f];	// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers * m_size.width - m_nLayers, ePot);
					} // if x, y

					if ((x < m_size.width - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x + 1) + f];	// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers * m_size.width + m_nLayers, ePot);
					} // x, y
				} // edges_diag
			} // x
#ifdef ENABLE_PPL
		}); // y
#else
		} // y
#endif
	}

	void CGraphLayered::fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const vec_mat_t &featureVectors, float *params, size_t params_len, float edgeWeight, float linkWeight)
	{
		const word	nFeatures	=static_cast<word>(featureVectors.size());

		// Assertions
		DGM_ASSERT(m_size.height == featureVectors[0].rows);
		DGM_ASSERT(m_size.width == featureVectors[0].cols);
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == getNumNodes());

#ifdef ENABLE_PPL
		concurrency::parallel_for(0, m_size.height, [&, nFeatures](int y) {
			Mat featureVector1(nFeatures, 1, CV_8UC1);
			Mat featureVector2(nFeatures, 1, CV_8UC1);
			Mat ePot;
			word l;
#else 
		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);
		Mat ePot;
		word l;
		for (int y = 0; y < m_size.height; y++) {
#endif
			byte const **pFv1 = new const byte * [nFeatures];
			for (word f = 0; f < nFeatures; f++) pFv1[f] = featureVectors[f].ptr<byte>(y);
			byte const **pFv2 = NULL;
			if (y > 0) {
				pFv2 = new const byte *[nFeatures];
				for (word f = 0; f < nFeatures; f++) pFv2[f] = featureVectors[f].ptr<byte>(y-1);
			}
			
			for (int x = 0; x < m_size.width; x++) {
				size_t idx = (y * m_size.width + x) * m_nLayers;
				
				for (word f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[f][x];				// featureVectors[x][y]

				if (m_gType & GRAPH_EDGES_LINK) {
					ePot = linkTrainer->getLinkPotentials(featureVector1, linkWeight);
					add(ePot, ePot.t(), ePot);
					if (m_nLayers >= 2)
						setArc(idx, idx + 1, ePot);
					ePot = CTrainEdgePotts::getEdgePotentials(100, m_nStates);
					for (l = 2; l < m_nLayers; l++)
						setEdge(idx + l - 1, idx + l, ePot);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[f][x - 1];				// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x];					// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers * m_size.width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x - 1];				// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers * m_size.width - m_nLayers, ePot);
					} // if x, y

					if ((x < m_size.width - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x + 1];				// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArc(idx + l, idx + l - m_nLayers * m_size.width + m_nLayers, ePot);
					} // x, y
				} // edges_diag
			} // x
#ifdef ENABLE_PPL
		}); // y
#else
		} // y
#endif
	}

	void CGraphLayered::defineEdgeGroup(float A, float B, float C, byte group)
	{
		// Assertion
		DGM_ASSERT_MSG(A != 0 || B != 0, "Wrong arguments");

#ifdef ENABLE_PPL
		concurrency::parallel_for(0, m_size.height, [&](int y) {
#else
		for (int y = 0; y < m_size.height; y++) {
#endif
			for (int x = 0; x < m_size.width; x++) {
				int i = (y * m_size.width + x) * m_nLayers;							// index of the current node from the base layer	
				int s = SIGN(A * x + B * y + C);									// sign of the current pixel according to the given line

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						int _x = x - 1;
						int _y = y;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) setArcGroup(i, i - m_nLayers, group);
					} // if x
					if (y > 0) {
						int _x = x;
						int _y = y - 1;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) setArcGroup(i, i - m_nLayers * m_size.width, group);
					} // if y
				}

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						int _x = x - 1;
						int _y = y - 1;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) setArcGroup(i, i - m_nLayers * m_size.width - m_nLayers, group);
					} // if x, y
					if ((x < m_size.width - 1) && (y > 0)) {
						int _x = x + 1;
						int _y = y - 1;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) setArcGroup(i, i - m_nLayers * m_size.width + m_nLayers, group);
					} // x, y
				}
			} // x
		} // y
#ifdef ENABLE_PPL
		);
#endif
	}

	void CGraphLayered::setGroupPot(byte group, const Mat &pot)
	{
		if (false) {
			for (int y = 0; y < m_size.height; y++) {
				for (int x = 0; x < m_size.width; x++) {
					int i = (y * m_size.width + x) * m_nLayers;							// index of the current node from the base layer	
					if (m_gType & GRAPH_EDGES_GRID) {
						if (x > 0) {
							if (getEdgeGroup(i, i - m_nLayers) == group)
								setArc(i, i - m_nLayers, pot);
						}
						if (y > 0) {
							if (getEdgeGroup(i, i - m_nLayers * m_size.width) == group)
								setArc(i, i - m_nLayers * m_size.width, pot);
						}
					}
					if (m_gType & GRAPH_EDGES_DIAG) {
						if ((x > 0) && (y > 0)) {
							if (getEdgeGroup(i, i - m_nLayers * m_size.width - m_nLayers) == group)
								setArc(i, i - m_nLayers * m_size.width - m_nLayers, pot);
						}
						if ((x < m_size.width - 1) && (y > 0)) {
							if (getEdgeGroup(i, i - m_nLayers * m_size.width + m_nLayers) == group)
								setArc(i, i - m_nLayers * m_size.width + m_nLayers, pot);
						}
					}
				} // x
			} // y
		}
		else {
			const size_t nNodes = getNumNodes();
			for (size_t n = 0; n < nNodes; n++) {
				vec_size_t vChilds;
				getChildNodes(n, vChilds);
				for (size_t c : vChilds) {
					if (getEdgeGroup(n, c) == group)
						setEdge(n, c, pot);
				}
			}
		}
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
