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

		void sigmoidFunction(Mat& X)
		{
			for (int y = 0; y < X.rows; y++){
				float* pX = X.ptr<float>(y);
				for (int x = 0; x < X.cols; x++)
					pX[x] = sigmoidFunction(pX[x]);
			}
		}
	}

	void CNeuronLayerMat::generateRandomWeights()
	{
        m_weights = random::U(m_weights.size(), m_weights.type(), -0.5f, 0.5f);
	}

	void CNeuronLayerMat::dotProd(const CNeuronLayerMat& layer)
	{
        parallel::gemm(layer.m_weights.t(), layer.m_values, 1, Mat(), 0, m_values);
		//sigmoidFunction(m_values);
	}

    void CNeuronLayerMat::applySigmoid(void)
    {
        sigmoidFunction(m_values);
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
		Mat DeltaIn_j;
        gemm(layerB.m_weights, resultErrorRate, 1, Mat(), 0, DeltaIn_j);

        // layerB.m_weights += learningRate * layerB.m_values x resultErrorRate.t()
        gemm(layerB.m_values, resultErrorRate.t(), learningRate, layerB.m_weights, 1, layerB.m_weights);

		//layerA.m_weights += learningRate * layerA.m_values x DeltaInJ.t();
        gemm(layerA.m_values, DeltaIn_j.t(), learningRate, layerA.m_weights, 1, layerA.m_weights);
	}
}}
