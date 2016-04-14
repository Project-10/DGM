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
						addArk(idx + l, idx + l + 1);
					//for (l = nLayers - 1; l < m_nLayers - 1; l++)
					//	addEdge(idx + l, idx + l + 1);
				}

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0)
						for (l = 0; l < m_nLayers; l++)
							addArk(idx + l, idx + l - m_nLayers);

					if (y > 0)
						for (l = 0; l < m_nLayers; l++)
							addArk(idx + l, idx + l - m_nLayers * graphSize.width);
				}

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							addArk(idx + l, idx + l - m_nLayers * graphSize.width - m_nLayers);

					if ((x < graphSize.width - 1) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							addArk(idx + l, idx + l - m_nLayers * graphSize.width + m_nLayers);
				}
			} // x
	}

	void CGraphLayered::fillNodes(const CTrainNode *nodeTrainerBase, const CTrainNode *nodeTrainerOccl, const Mat &featureVectors, float weightBase, float weightOccl)
	{
		int height = featureVectors.rows;
		int width = featureVectors.cols;
		int nFeatures = featureVectors.channels();

		// Assertions
		DGM_ASSERT(nFeatures == nodeTrainerBase->getNumFeatures());
		if (nodeTrainerOccl) DGM_ASSERT(nFeatures == nodeTrainerOccl->getNumFeatures());
		DGM_ASSERT(width * height * m_nLayers == getNumNodes());

		Rect ROIb(0, 0, 1, nodeTrainerBase->getNumStates());
		Rect ROIo(0, m_nStates - nodeTrainerOccl->getNumStates(), 1, nodeTrainerOccl->getNumStates());

#ifdef USE_PPL
		concurrency::parallel_for(0, height, [&, width, nFeatures, ROIb, ROIo](int y) {
			Mat featureVector(nFeatures, 1, CV_8UC1);
			Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
			const byte *pFv = featureVectors.ptr<byte>(y);
			for (int x = 0, i = y * width * m_nLayers; x < width; x++, i += m_nLayers) {
				for (byte f = 0; f < nFeatures; f++) featureVector.at<byte>(f, 0) = pFv[nFeatures * x + f];			// featureVectors[x][y]
				nodeTrainerBase->getNodePotentials(featureVector, weightBase).copyTo(nPotBase(ROIb));
				setNode(i, nPotBase);
				if (m_nLayers >= 2) {
					nodeTrainerOccl->getNodePotentials(featureVector, weightOccl).copyTo(nPotOccl(ROIo));
					setNode(i + 1, nPotOccl);
				}
			} // x	
		}); // y
#else
		Mat featureVector(nFeatures, 1, CV_8UC1);
		Mat nPotBase(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		Mat nPotOccl(m_nStates, 1, CV_32FC1, Scalar(0.0f));
		for (int y = 0, i = 0; y < height; y++) {
			const byte *pFv = featureVectors.ptr<byte>(y);
			for (int x = 0; x < width; x++, i += m_nLayers) {
				for (byte f = 0; f < nFeatures; f++) featureVector.at<byte>(f, 0) = pFv[nFeatures * x + f];				// featureVectors[x][y]
				nodeTrainerBase->getNodePotentials(featureVector, weightBase).copyTo(nPotBase(ROIb));
				setNode(i, nPotBase);
				if (m_nLayers >= 2) {
					nodeTrainerOccl->getNodePotentials(featureVector, weightOccl).copyTo(nPotOccl(ROIo));
					setNode(i + 1, nPotOccl);
				}
			} // x	
		} // y
#endif
	}

	void CGraphLayered::fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const Mat &featureVectors, float *params, size_t params_len, float edgeWeight, float linkWeight)
	{
		int height = featureVectors.rows;
		int width = featureVectors.cols;
		int nFeatures = featureVectors.channels();

		// Assertions
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(width * height * m_nLayers == getNumNodes());

#ifdef USE_PPL
		concurrency::parallel_for(0, height, [&, width, nFeatures](int y)
		{
			Mat featureVector1(nFeatures, 1, CV_8UC1);
			Mat featureVector2(nFeatures, 1, CV_8UC1);
			Mat ePot;
			size_t l;
			const byte *pFv1 = featureVectors.ptr<byte>(y);
			const byte *pFv2 = (y > 0) ? featureVectors.ptr<byte>(y - 1) : NULL;
			for (int x = 0, i = y * width * m_nLayers; x < width; x++, i += m_nLayers) {
				for (byte f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];				// featureVectors[x][y]
				
				if (m_gType & GRAPH_EDGES_LINK) {
					ePot = linkTrainer->getLinkPotentials(featureVector1, linkWeight);
					add(ePot, ePot.t(), ePot);
					word nLayers = MIN(2, m_nLayers);
					for (l = 0; l < nLayers - 1; l++)
						setArk(i + l, i + l + 1, ePot);
					//for (l = nLayers - 1; l < m_nLayers - 1; l++)
					//	setEdge(i + l, i + l + 1);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers * width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x - 1) + f];	// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers * width - m_nLayers, ePot);
					} // if x, y

					if ((x < width - 1) && (y > 0)) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x + 1) + f];	// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers * width + m_nLayers, ePot);
					} // x, y
				} // edges_diag
			} // x
		}); // y
#else
		Mat featureVector1(nFeatures, 1, CV_8UC1);
		Mat featureVector2(nFeatures, 1, CV_8UC1);
		Mat ePot;
		word l;
		for (int y = 0, i = 0; y < height; y++) {
			const byte *pFv1 = featureVectors.ptr<byte>(y);
			const byte *pFv2 = (y > 0) ? featureVectors.ptr<byte>(y - 1) : NULL;
			for (int x = 0; x < width; x++, i += m_nLayers) {
				for (byte f = 0; f < nFeatures; f++) featureVector1.at<byte>(f, 0) = pFv1[nFeatures * x + f];				// featureVectors[x][y]
				
				if (m_gType & GRAPH_EDGES_LINK) {
					word nLayers = MIN(2, m_nLayers);
					for (l = 0; l < nLayers - 1; l++)
						setArk(i + l, i + l + 1, linkTrainer->getLinkPotentials(featureVector1, linkWeight));
					//for (l = nLayers - 1; l < m_nLayers - 1; l++)
					//	setEdge(i + l, i + l + 1);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers * width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x - 1) + f];	// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers * width - m_nLayers, ePot);
					} // if x, y
				
					if ((x < width - 1) && (y > 0)) {
						for (byte f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x + 1) + f];	// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) setArk(i + l, i + l - m_nLayers * width + m_nLayers, ePot);
					} // x, y
				} // edges_diag

			} // x
		} // y
#endif
	}
	
	/// @todo: Implement edge potential transfer
	void CGraphLayered::marginalize(size_t node)
	{
		// search all incoming edges
		vec_size_t parentNodes, childNodes;
		getParentNodes(node, parentNodes);
		getChildNodes(node, childNodes);
		
		for (size_t &parent : parentNodes) {

			// Filter those incoming edges, which are parts of arcs
			auto isArk = std::find(childNodes.begin(), childNodes.end(), parent);
			if (isArk != childNodes.end()) continue;

			// here all incoming edges are direct edges
			
			// Connect with arks
			for (size_t &parent2 : parentNodes) {
				if (parent == parent2) continue;

				auto isArk = std::find(childNodes.begin(), childNodes.end(), parent2);
				if (isArk != childNodes.end()) continue;

				addArk(parent, parent2);
			}
			
			// Connect to childs with edges
			for (size_t &child : childNodes) addEdge(parent, child);
		}

		// Delete all	
		for (size_t &parent : parentNodes) removeEdge(parent, node);
		for (size_t &child : childNodes)  removeEdge(node, child);
	}

	/// @todo: Implement this function
	void CGraphLayered::marginalize(vec_size_t nodes)
	{




	}

}