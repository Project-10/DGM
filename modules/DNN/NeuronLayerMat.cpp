#include "NeuronLayerMat.h"
#include "DGM/random.h"
#include "DGM/parallel.h"
#include "macroses.h"

namespace DirectGraphicalModels { namespace dnn
{
	namespace {
		/**
		 * Applies the Sigmoid Activation function
		 *
		 * @param the value at each node
		 * @return a number between 0 and 1.
		 */
		float sigmoidFunction(float x)
		{
			return 1.0f / (1.0f + expf(-x));
		}
	}
	
	void CNeuronLayerMat::generateRandomWeights(void)
	{
		m_weights = random::U(m_weights.size(), m_weights.type(), -0.5f, 0.5f);
	}


	void CNeuronLayerMat::dotProd(const CNeuronLayerMat& layer)
	{
        Mat AA = layer.getValues();
        Mat BB = layer.getWeights();
        Mat res = BB.t() * AA;
        for(int i=0; i < m_values.rows; i++){
            float x = sigmoidFunction(res.at<float>(0,i));
            m_values.at<float>(0,i) = x;
        }
	}

	void CNeuronLayerMat::setValues(const Mat& values)
	{
		// Assertions
		DGM_ASSERT(values.type() == m_values.type());
		DGM_ASSERT(values.size() == m_values.size());
		values.copyTo(m_values);
	}

	void CNeuronLayerMat::backPropagate(CNeuronLayerMat& layerA, CNeuronLayerMat& layerB, CNeuronLayerMat& layerC, const Mat& resultErrorRate, float learningRate)
	{
        //  Mat layerB_weights = layerB.getWeights(); // 60 x 10
        //  Mat layerB_values = layerB.getValues();   // 60 x 1
        //  Mat layerA_weights = layerA.getValues();  // 784 x 1

        Mat DeltaWjk(layerB.getNumNeurons(), layerC.getNumNeurons(), CV_32FC1);
        Mat DeltaVjk(layerA.getNumNeurons(), layerB.getNumNeurons(), CV_32FC1);
        Mat DeltaIn_j(layerB.getNumNeurons(), 1, CV_32FC1);
        Mat DeltaJ(layerB.getNumNeurons(), 1, CV_32FC1);
        
        DeltaWjk  = learningRate * layerB.getValues() * resultErrorRate.t(); // 60 x 10
        DeltaIn_j = layerB.getWeights() * resultErrorRate;
         
        for(int i=0; i < layerB.getNumNeurons(); i++){
            float sigmoid = sigmoidFunction(layerB.m_values.at<float>(i, 0));
            DeltaJ.at<float>(i,0) = DeltaIn_j.at<float>(i,0) * sigmoid * (1-sigmoid);
        }
        
        DeltaVjk = learningRate * layerA.getValues() * DeltaJ.t();
        
        for(int i = 0; i < layerA.getNumNeurons(); i++) {
            for(int j = 0; j < layerB.getNumNeurons(); j++) {
                float oldWeight = layerA.m_weights.at<float>(i, j);
                layerA.m_weights.at<float>(i, j) = oldWeight + DeltaVjk.at<float>(i,j);
            }
        }
        
        for(int i = 0; i < layerB.getNumNeurons(); i++) {
            for(int j = 0; j < layerC.getNumNeurons(); j++) {
                float oldWeight = layerB.m_weights.at<float>(i, j);
                layerB.m_weights.at<float>(i, j) = oldWeight + DeltaWjk.at<float>(i,j);
            }
        }
	}
}}
