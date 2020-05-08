#include "TestParamEstimation.h"
#include "DGM/random.h"

// TODO: test with empty parameters ?
void CTestParamEstimation::testParamEstimation(CParamEstimation& paramEstimator)
{
	for (size_t i = 0; i < nParams; i++) {
		m_vInitParams[i] = random::U<float>(-10, 10);
		m_vInitDeltas[i] = 1e-4f;						// accuracy
		m_vSolution[i]	 = random::U<float>(-10, 10);
	}

	vec_float_t vParams = m_vInitParams;
	paramEstimator.setInitParams(m_vInitParams);
	paramEstimator.setDeltas(m_vInitDeltas);

	while (!paramEstimator.isConverged()) {
		float val = objectiveFunction(vParams);
		vParams = paramEstimator.getParams(val);
	}

	//printf("Solution:");
	//for (float p : m_vSolution) printf("%.2f ", p);
	//printf("\n");
	//printf("Optimal parameters: ");
	//for (float p : vParams) printf("%.2f ", p);
	//printf("\n");

	// Check result
	for (size_t i = 0; i < nParams; i++) 
		ASSERT_GE(m_vInitDeltas[i], fabs(vParams[i] - m_vSolution[i]));
}

float CTestParamEstimation::objectiveFunction(const vec_float_t& vParams)
{
	float res = 0;
	for (size_t i = 0; i < vParams.size(); i++)
		res += fabs(vParams[i] - m_vSolution[i]);
	res = 10 * vParams.size() - res;
	if (res < 0) res = 0;

	return res;
}


TEST_F(CTestParamEstimation, Powell) 
{
	CParamEstimationPowell powell(nParams);
	testParamEstimation(powell);
}

TEST_F(CTestParamEstimation, PSO)
{
	CParamEstimationPSO pso(nParams);
	testParamEstimation(pso); // TODO: uncomment
}
