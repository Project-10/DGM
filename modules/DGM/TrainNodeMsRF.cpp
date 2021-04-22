#include "TrainNodeMsRF.h"

#ifdef USE_SHERWOOD

#include "sherwood/Sherwood.h"

#ifdef ENABLE_PDP
#include "sherwood/ParallelForestTrainer.h"				// for parallle computing
#endif

#include "sherwood/utilities/FeatureResponseFunctions.h"
#include "sherwood/utilities/StatisticsAggregators.h"
#include "sherwood/utilities/DataPointCollection.h"
#include "sherwood/utilities/TrainingContexts.h"

namespace DirectGraphicalModels
{
// Constructor
    CTrainNodeMsRF::CTrainNodeMsRF(byte nStates, word nFeatures, TrainNodeMsRFParams params) : CBaseRandomModel(nStates), CTrainNode(nStates, nFeatures)
{
	init(params);
}

// Constructor
    CTrainNodeMsRF::CTrainNodeMsRF(byte nStates, word nFeatures, size_t maxSamples) : CBaseRandomModel(nStates), CTrainNode(nStates, nFeatures)
{
	TrainNodeMsRFParams params = TRAIN_NODE_MS_RF_PARAMS_DEFAULT;
	params.maxSamples = maxSamples;
	init(params);
}

void CTrainNodeMsRF::init(TrainNodeMsRFParams params)
{
    m_pSamplesAcc	= std::unique_ptr<CSamplesAccumulator>(new CSamplesAccumulator(m_nStates, params.maxSamples));
	m_pParams		= std::unique_ptr<sw::TrainingParameters>(new sw::TrainingParameters());
	// Some default parameters
	m_pParams->MaxDecisionLevels						= params.max_decision_levels - 1;
	m_pParams->NumberOfCandidateFeatures				= params.num_of_candidate_features;
	m_pParams->NumberOfCandidateThresholdsPerFeature	= params.num_of_candidate_thresholds_per_feature;
	m_pParams->NumberOfTrees							= params.num_ot_trees;
	m_pParams->Verbose									= params.verbose;
}

// Destructor
CTrainNodeMsRF::~CTrainNodeMsRF(void)
{}

void CTrainNodeMsRF::reset(void) 
{
	m_pSamplesAcc->reset();
	m_pRF.reset();
}

void CTrainNodeMsRF::save(const std::string &path, const std::string &name, short idx) const
{
	std::string fileName = generateFileName(path, name.empty() ?  "TrainNodeMsRF" : name, idx);
	m_pRF->Serialize(fileName);
}

void CTrainNodeMsRF::load(const std::string &path, const std::string &name, short idx)
{
	std::string fileName = generateFileName(path, name.empty() ?  "TrainNodeMsRF" : name, idx);
    m_pRF = sw::Forest<sw::LinearFeatureResponse, sw::HistogramAggregator>::Deserialize(fileName);
}

void CTrainNodeMsRF::addFeatureVec(const Mat &featureVector, byte gt)
{
	m_pSamplesAcc->addSample(featureVector, gt);
}

void CTrainNodeMsRF::train(bool doClean)
{
#ifdef DEBUG_PRINT_INFO
	printf("\n");
#endif
	// Filling <pData>
	sw::DataPointCollection * pData = new sw::DataPointCollection();
	pData->m_dimension = getNumFeatures();

	for (byte s = 0; s < m_nStates; s++) {						// states
		int nSamples = m_pSamplesAcc->getNumSamples(s);
#ifdef DEBUG_PRINT_INFO		
		printf("State[%d] - %d of %d samples\n", s, nSamples, m_pSamplesAcc->getNumInputSamples(s));
#endif
		for (int smp = 0; smp < nSamples; smp++) {
			for (word f = 0; f < getNumFeatures(); f++) {			// features
				byte fval = m_pSamplesAcc->getSamplesContainer(s).at<byte>(smp, f);
				pData->m_vData.push_back(fval);
			} // f
			pData->m_vLabels.push_back(s);
		} // smp
		if (doClean) m_pSamplesAcc->release(s);					// releases memory
	} // s


	// Training
	sw::Random random;
	sw::ClassificationTrainingContext classificationContext(m_nStates, getNumFeatures());
#ifdef ENABLE_PDP
	// Use this function with cautions - it is not verifiied!
	m_pRF = sw::ParallelForestTrainer<sw::LinearFeatureResponse, sw::HistogramAggregator>::TrainForest(random, *m_pParams, classificationContext, *pData);
#else
	m_pRF = sw::ForestTrainer<sw::LinearFeatureResponse, sw::HistogramAggregator>::TrainForest(random, *m_pParams, classificationContext, *pData);
#endif

	delete pData;
}	

void CTrainNodeMsRF::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	std::unique_ptr<sw::DataPointCollection> testData = std::unique_ptr<sw::DataPointCollection>(new sw::DataPointCollection());
	testData->m_dimension = getNumFeatures();
	for (word f = 0; f < getNumFeatures(); f++) {
		float feature = static_cast<float>(featureVector.ptr<byte>(f)[0]);
		testData->m_vData.push_back(feature);
	}

	std::vector<std::vector<int>> leafNodeIndices;
	m_pRF->Apply(*testData, leafNodeIndices);
	
	sw::HistogramAggregator h(m_nStates);
	int index = 0;
	for (size_t t = 0; t < m_pRF->TreeCount(); t++) {
		int leafIndex = leafNodeIndices[t][index];
		h.Aggregate(m_pRF->GetTree((t)).GetNode(leafIndex).TrainingDataStatistics);
	} // t

	float mudiness = static_cast<float> (0.5 * h.Entropy());

	for (byte s = 0; s < m_nStates; s++) 
		potential.at<float>(s, 0) = (1.0f - mudiness) * h.GetProbability(s);
}
}
#endif
