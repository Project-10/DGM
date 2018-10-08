#include "GraphLayered.h"
#include "GraphPairwise.h"

#include "TrainNode.h"
#include "TrainEdge.h"
#include "TrainEdgePotts.h"
#include "TrainLink.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
	void CGraphLayered::addNodes(CvSize graphSize)
	{
		if (m_graph.getNumNodes() != 0) m_graph.reset();
		m_size = graphSize;

		word l;
		for (int y = 0; y < m_size.height; y++)
			for (int x = 0; x < m_size.width; x++) {
				// Nodes
				size_t idx = m_graph.addNode();
				for (l = 1; l < m_nLayers; l++) m_graph.addNode();

				if (m_gType & GRAPH_EDGES_LINK) {
					if (m_nLayers >= 2)
						m_graph.addArc(idx, idx + 1);
					for (l = 2; l < m_nLayers; l++)
						m_graph.addEdge(idx + l - 1, idx + l);
				} // if LINK

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0)
						for (l = 0; l < m_nLayers; l++)
							m_graph.addArc(idx + l, idx + l - m_nLayers);
					if (y > 0)
						for (l = 0; l < m_nLayers; l++)
							m_graph.addArc(idx + l, idx + l - m_nLayers * m_size.width);
				} // if GRID
			} // x


		if (m_gType & GRAPH_EDGES_DIAG) {
			for (int y = 0; y < m_size.height; y++) {
				for (int x = 0; x < m_size.width; x++) {
					size_t idx = (y * m_size.width + x) * m_nLayers;

					if ((x > 0) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							m_graph.addArc(idx + l, idx + l - m_nLayers * (m_size.width + 1));

					if ((x < graphSize.width - 1) && (y > 0))
						for (l = 0; l < m_nLayers; l++)
							m_graph.addArc(idx + l, idx + l - m_nLayers * (m_size.width - 1));
				} // x
			} // y
		} // if DIAG
	}

	void CGraphLayered::setNodes(const Mat &potBase, const Mat &potOccl)
	{
		// Assertions
		DGM_ASSERT(m_size.height == potBase.rows);
		DGM_ASSERT(m_size.width == potBase.cols);
		DGM_ASSERT(CV_32F == potBase.depth());
		if (!potOccl.empty()) {
			DGM_ASSERT(potBase.size() == potOccl.size());
			DGM_ASSERT(CV_32F == potOccl.depth());
		}
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == m_graph.getNumNodes());

		byte nStatesBase = static_cast<byte>(potBase.channels());
		byte nStatesOccl = potOccl.empty() ? 0 : static_cast<byte>(potOccl.channels());
		if (m_nLayers >= 2) DGM_ASSERT(nStatesOccl);
		DGM_ASSERT(nStatesBase + nStatesOccl == m_graph.getNumStates());

#ifdef ENABLE_PPL
		concurrency::parallel_for(0, m_size.height, [&, nStatesBase, nStatesOccl](int y) {
			Mat nPotBase(m_graph.getNumStates(), 1, CV_32FC1, Scalar(0.0f));
			Mat nPotOccl(m_graph.getNumStates(), 1, CV_32FC1, Scalar(0.0f));
			Mat nPotIntr(m_graph.getNumStates(), 1, CV_32FC1, Scalar(0.0f));
			for (byte s = 0; s < nStatesOccl; s++)
				nPotIntr.at<float>(m_graph.getNumStates() - nStatesOccl + s, 0) = 100.0f / nStatesOccl;
#else
		Mat nPotBase(m_graph.getNumStates(), 1, CV_32FC1, Scalar(0.0f));
		Mat nPotOccl(m_graph.getNumStates(), 1, CV_32FC1, Scalar(0.0f));
		Mat nPotIntr(m_graph.getNumStates(), 1, CV_32FC1, Scalar(0.0f));
		for (byte s = 0; s < nStatesOccl; s++) 
			nPotIntr.at<float>(m_graph.getNumStates() - nStatesOccl + s, 0) = 100.0f / nStatesOccl;
		for (int y = 0; y < m_size.height; y++) {
#endif
			const float *pPotBase = potBase.ptr<float>(y);
			const float *pPotOccl = potOccl.empty() ? NULL : potOccl.ptr<float>(y);
			for (int x = 0; x < m_size.width; x++) {
				size_t idx = (y * m_size.width + x) * m_nLayers;
				
				for (byte s = 0; s < nStatesBase; s++) 
					nPotBase.at<float>(s, 0) = pPotBase[nStatesBase * x + s];
				m_graph.setNode(idx, nPotBase);
				
				if (m_nLayers >= 2) {
					for (byte s = 0; s < nStatesOccl; s++) 
						nPotOccl.at<float>(m_graph.getNumStates() - nStatesOccl + s, 0) = pPotOccl[nStatesOccl * x + s];
					m_graph.setNode(idx + 1, nPotOccl);
				}
				
				for (word l = 2; l < m_nLayers; l++)
					m_graph.setNode(idx + l, nPotIntr);
			} // x
		} // y
#ifdef ENABLE_PPL	
		);
#endif
	}

	void CGraphLayered::fillEdges(const CTrainEdge *edgeTrainer, const CTrainLink *linkTrainer, const Mat &featureVectors, float *params, size_t params_len, float edgeWeight, float linkWeight)
	{
		const word	nFeatures	= featureVectors.channels();

		// Assertions
		DGM_ASSERT(m_size.height == featureVectors.rows);
		DGM_ASSERT(m_size.width == featureVectors.cols);
		DGM_ASSERT(edgeTrainer);
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == m_graph.getNumNodes());

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
						m_graph.setArc(idx, idx + 1, ePot);
					ePot = CTrainEdge::getDefaultEdgePotentials(100, m_graph.getNumStates());
					for (l = 2; l < m_nLayers; l++)
						m_graph.setEdge(idx + l - 1, idx + l, ePot);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[nFeatures * (x - 1) + f];	// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * x + f];		// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers * m_size.width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x - 1) + f];	// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers * m_size.width - m_nLayers, ePot);
					} // if x, y

					if ((x < m_size.width - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[nFeatures * (x + 1) + f];	// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers * m_size.width + m_nLayers, ePot);
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
		DGM_ASSERT(edgeTrainer);
		DGM_ASSERT(nFeatures == edgeTrainer->getNumFeatures());
		if (linkTrainer) DGM_ASSERT(nFeatures == linkTrainer->getNumFeatures());
		DGM_ASSERT(m_size.width * m_size.height * m_nLayers == m_graph.getNumNodes());

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
						m_graph.setArc(idx, idx + 1, ePot);
					ePot = CTrainEdge::getDefaultEdgePotentials(100, m_graph.getNumStates());
					for (l = 2; l < m_nLayers; l++)
						m_graph.setEdge(idx + l - 1, idx + l, ePot);
				} // edges_link

				if (m_gType & GRAPH_EDGES_GRID) {
					if (x > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv1[f][x - 1];				// featureVectors[x-1][y]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers, ePot);
					} // if x

					if (y > 0) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x];					// featureVectors[x][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers * m_size.width, ePot);
					} // if y
				} // edges_grid

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x - 1];				// featureVectors[x-1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers * m_size.width - m_nLayers, ePot);
					} // if x, y

					if ((x < m_size.width - 1) && (y > 0)) {
						for (word f = 0; f < nFeatures; f++) featureVector2.at<byte>(f, 0) = pFv2[f][x + 1];				// featureVectors[x+1][y-1]
						ePot = edgeTrainer->getEdgePotentials(featureVector1, featureVector2, params, params_len, edgeWeight);
						for (word l = 0; l < m_nLayers; l++) m_graph.setArc(idx + l, idx + l - m_nLayers * m_size.width + m_nLayers, ePot);
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
						if (s != _s) m_graph.setArcGroup(i, i - m_nLayers, group);
					} // if x
					if (y > 0) {
						int _x = x;
						int _y = y - 1;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) m_graph.setArcGroup(i, i - m_nLayers * m_size.width, group);
					} // if y
				}

				if (m_gType & GRAPH_EDGES_DIAG) {
					if ((x > 0) && (y > 0)) {
						int _x = x - 1;
						int _y = y - 1;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) m_graph.setArcGroup(i, i - m_nLayers * m_size.width - m_nLayers, group);
					} // if x, y
					if ((x < m_size.width - 1) && (y > 0)) {
						int _x = x + 1;
						int _y = y - 1;
						int _s = SIGN(A * _x + B * _y + C);
						if (s != _s) m_graph.setArcGroup(i, i - m_nLayers * m_size.width + m_nLayers, group);
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
							if (m_graph.getEdgeGroup(i, i - m_nLayers) == group)
								m_graph.setArc(i, i - m_nLayers, pot);
						}
						if (y > 0) {
							if (m_graph.getEdgeGroup(i, i - m_nLayers * m_size.width) == group)
								m_graph.setArc(i, i - m_nLayers * m_size.width, pot);
						}
					}
					if (m_gType & GRAPH_EDGES_DIAG) {
						if ((x > 0) && (y > 0)) {
							if (m_graph.getEdgeGroup(i, i - m_nLayers * m_size.width - m_nLayers) == group)
								m_graph.setArc(i, i - m_nLayers * m_size.width - m_nLayers, pot);
						}
						if ((x < m_size.width - 1) && (y > 0)) {
							if (m_graph.getEdgeGroup(i, i - m_nLayers * m_size.width + m_nLayers) == group)
								m_graph.setArc(i, i - m_nLayers * m_size.width + m_nLayers, pot);
						}
					}
				} // x
			} // y
		}
		else {
			const size_t nNodes = m_graph.getNumNodes();
			for (size_t n = 0; n < nNodes; n++) {
				vec_size_t vChilds;
				m_graph.getChildNodes(n, vChilds);
				for (size_t c : vChilds) {
					if (m_graph.getEdgeGroup(n, c) == group)
						m_graph.setEdge(n, c, pot);
				}
			}
		}
	}
}
