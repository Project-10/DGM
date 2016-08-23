#include "TrainNodeMsRF.h"
#include "macroses.h"

#ifdef USE_SHERWOOD

#include "sherwood\Sherwood.h"

#ifdef ENABLE_PPL
#include "sherwood\ParallelForestTrainer.h"				// for parallle computing
#endif

#include "sherwood\utilities\FeatureResponseFunctions.h"
#include "sherwood\utilities\StatisticsAggregators.h"
#include "sherwood\utilities\DataPointCollection.h"
#include "sherwood\utilities\TrainingContexts.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainNodeMsRF::CTrainNodeMsRF(byte nStates, word nFeatures, TrainNodeMsRFParams params) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)//, pData(NULL), pForest(NULL)
{
	init(params);
}

// Constructor
CTrainNodeMsRF::CTrainNodeMsRF(byte nStates, word nFeatures, int maxSamples) : CTrainNode(nStates, nFeatures), CBaseRandomModel(nStates)//, pData(NULL), pForest(NULL)
{
	TrainNodeMsRFParams params = TRAIN_NODE_MS_RF_PARAMS_DEFAULT;
	params.maxSamples = maxSamples;
	init(params);
}

void CTrainNodeMsRF::init(TrainNodeMsRFParams params)
{
	// Some default parameters
	m_pParams = std::auto_ptr<sw::TrainingParameters>(new sw::TrainingParameters());
	m_pParams->MaxDecisionLevels						= params.max_decision_levels - 1;
	m_pParams->NumberOfCandidateFeatures				= params.num_of_candidate_features;
	m_pParams->NumberOfCandidateThresholdsPerFeature	= params.num_of_candidate_thresholds_per_feature;
	m_pParams->NumberOfTrees							= params.num_ot_trees;
	m_pParams->Verbose									= params.verbose;

	m_pData = std::auto_ptr<sw::DataPointCollection>(new sw::DataPointCollection());
	m_pData->m_dimension = m_nFeatures;

	if (params.maxSamples == 0) m_maxSamples = std::numeric_limits<dword>::max();
	else m_maxSamples = params.maxSamples;
}

// Destructor
CTrainNodeMsRF::~CTrainNodeMsRF(void)
{ }

void CTrainNodeMsRF::reset(void) 
{
	m_pData->m_vData.clear();
	m_pData->m_vLabels.clear();
	m_pForest.reset();
}

void CTrainNodeMsRF::save(const std::string &path, const std::string &name, short idx) const
{
	std::string fileName = generateFileName(path, name.empty() ?  "TrainNodeMsRF" : name, idx);
	m_pForest->Serialize(fileName);
}

void CTrainNodeMsRF::load(const std::string &path, const std::string &name, short idx)
{
	std::string fileName = generateFileName(path, name.empty() ?  "TrainNodeMsRF" : name, idx);
	m_pForest = sw::Forest<sw::LinearFeatureResponse, sw::HistogramAggregator>::Deserialize(fileName);
}

void CTrainNodeMsRF::addFeatureVec(const Mat &featureVector, byte gt) 
{
	// Assertions
	DGM_ASSERT_MSG(gt < m_nStates, "The groundtruth value %u is out of range [0; %u)", gt, m_nStates);
	DGM_ASSERT_MSG(featureVector.type() == CV_8UC1, "The feature vector has incorrect type");

	m_pData->m_vLabels.push_back(gt);
	
	for (word f = 0; f < m_nFeatures; f++) {
		byte fval = featureVector.at<byte>(f, 0);
		m_pData->m_vData.push_back(fval);
	}	
}	

void CTrainNodeMsRF::train(void) 
{
/*	size_t * toDel = new size_t[m_nStates];
	size_t * excessSamples = 0;
	for (byte s = 0; s < m_nStates; s++) {
		toDel[s] = MAX(0, static_cast<long>(m_pData->Count(s)) - static_cast<long>(m_maxSamples));
		excessSamples += toDel[s];
		printf("Class[%d] has %d samples. %d samples will be deleted\n", s, m_pData->Count(s), toDel[s]);
	}

	srand(static_cast<unsigned int>(time(NULL)));	
	dword allSamples = m_pData->Count();
	while (excessSamples > 0) {
		dRand() % allSamples;

		excessSamples--;
	}
	
	delete [] toDel;*/
	
	
	sw::Random random;
	sw::ClassificationTrainingContext classificationContext(m_nStates, m_nFeatures);
#ifdef ENABLE_PPL
	// Use this function with cautions - it is not verifiied!
	m_pForest = sw::ParallelForestTrainer<sw::LinearFeatureResponse, sw::HistogramAggregator>::TrainForest(random, *m_pParams, classificationContext, *m_pData);
#else
	m_pForest = sw::ForestTrainer<sw::LinearFeatureResponse, sw::HistogramAggregator>::TrainForest(random, *m_pParams, classificationContext, *m_pData);
#endif
}	

void CTrainNodeMsRF::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
	std::auto_ptr<sw::DataPointCollection> testData = std::auto_ptr<sw::DataPointCollection>(new sw::DataPointCollection());
	testData->m_dimension = m_nFeatures;
	for (word f = 0; f < m_nFeatures; f++) {
		float feature = static_cast<float>(featureVector.ptr<byte>(f)[0]);
		testData->m_vData.push_back(feature);
	}

	std::vector<std::vector<int>> leafNodeIndices;
	m_pForest->Apply(*testData, leafNodeIndices);
	
	sw::HistogramAggregator h(m_nStates);
	int index = 0;
	for (size_t t = 0; t < m_pForest->TreeCount(); t++) {
		int leafIndex = leafNodeIndices[t][index];
		h.Aggregate(m_pForest->GetTree((t)).GetNode(leafIndex).TrainingDataStatistics);
	} // t

	float mudiness = static_cast<float> (0.5 * h.Entropy());

	for (byte s = 0; s < m_nStates; s++) 
		potential.at<float>(s, 0) = (1.0f - mudiness) * h.GetProbability(s);
}
}
#endif
