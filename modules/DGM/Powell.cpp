#include "Powell.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CPowell::CPowell(word nParams) : m_nParams(nParams)
{
	m_pParam	 = std::make_unique<float[]>(nParams);
	m_pDelta	 = std::make_unique<float[]>(nParams);
	m_vMin		 = vec_float_t(nParams);
	m_vMax		 = vec_float_t(nParams);
	m_vConverged = vec_bool_t(nParams);
	m_vKappa	 = vec_float_t(3);
	reset();
}

// Destructor
CPowell::~CPowell(void)
{
}

void CPowell::reset(void)
{
	m_paramID		= 0;				// first parameter
	m_nSteps		= 0;			
	m_koeff			= 1.0f;				// identity koefficient
	m_acceleration  = 0.1f;				// default search acceleration

	std::fill_n(m_pParam.get(), m_nParams,  0.0f);
	std::fill_n(m_pDelta.get(), m_nParams,  0.1f);
	std::fill(m_vMin.begin(), m_vMin.end(), -FLT_MAX);
	std::fill(m_vMax.begin(), m_vMax.end(),  FLT_MAX);
	std::fill(m_vConverged.begin(), m_vConverged.end(), false);
	std::fill(m_vKappa.begin(), m_vKappa.end(), -1.0f);
}

void CPowell::setInitParams(float *pParam) 
{
	for (word p = 0; p < m_nParams; p++) {
		if (pParam[p] > m_vMax[p]) {
			DGM_WARNING("Argument[%d]=%.2f exceeds the upper boundary %.2f and will not be set", p, pParam[p], m_vMax[p]);
			continue;
		}
		if (pParam[p] < m_vMin[p]) {
			DGM_WARNING("Argument[%d]=%.2f exceeds the lower boundary %.2f and will not be set", p, pParam[p], m_vMin[p]);
			continue;
		}
		m_pParam[p] = pParam[p];
	}
}

void CPowell::setMinParams(float *pMinParam) 
{ 
	for (word p = 0; p < m_nParams; p++) 
		if (pMinParam[p] > m_pParam[p]) DGM_WARNING("pMinParam[%d]=%.2f contradicts the parameter value %.2f and will not be set", p, pMinParam[p], m_pParam[p]);
		else m_vMin[p] = pMinParam[p];
}

void CPowell::setMaxParams(float *pMaxParam) 
{ 
	for (word p = 0; p < m_nParams; p++) 
		if (pMaxParam[p] < m_pParam[p]) DGM_WARNING("pParam[%d]=%.2f contradicts the parameter value %.2f and will not be set", p, pMaxParam[p], m_pParam[p]);
		else m_vMax[p] = pMaxParam[p];
}

void CPowell::setDeltas(float *pDelta) 
{ 
	memcpy(m_pDelta.get(), pDelta, m_nParams * sizeof(float)); 
}

void CPowell::setAcceleration(float acceleration)
{
	if (acceleration >= 0.0f) m_acceleration = acceleration;
	else DGM_WARNING("Negative acceleration value was not set");
}

float * CPowell::getParams(float kappa)
{
	// Assertions
	DGM_ASSERT_MSG(kappa > 0.0f, "Negative kappa values are not allowed");

#ifdef DEBUG_PRINT_INFO
	// Printing out the information
	printf("[%d]:\t", m_paramID);
	for (word i = 0; i < m_nParams; i++) printf("%.2f\t", m_pParam[i]);
	printf("%.2f\n", kappa);
#endif

	// If converged, no further steps are required
	if (isConverged()) return m_pParam.get();

	// =============== Fill all 3 kappa values ===============
		 if (m_vKappa[oD] < 0) { m_vKappa[oD] = kappa; m_midPoint = curArg; } 
	else if (m_vKappa[mD] < 0)   m_vKappa[mD] = kappa;
	else if (m_vKappa[pD] < 0)   m_vKappa[pD] = kappa;

	while (true) {

		// Need kappa: -1
		if (m_vKappa[mD] < 0) {
			if (m_midPoint == minArg) m_vKappa[mD] = 0.0f;
			else {
				curArg = MAX(minArg, m_midPoint - m_koeff * delta);
				return m_pParam.get();
			}
		}

		// Need kappa: +1
		if (m_vKappa[pD] < 0) {
			if (m_midPoint == maxArg) m_vKappa[pD] = 0.0f;
			else {
				curArg = MIN(maxArg, m_midPoint + m_koeff * delta);
				return m_pParam.get();
			}
		}

		// =============== All 3 kappas are ready ===============
		float maxKappa = *std::max_element(m_vKappa.begin(), m_vKappa.end());

		if (maxKappa == m_vKappa[oD]) {			// >>>>> Middle value -> Proceed to the next argument
			convArg = true;
			curArg = m_midPoint;

			if (isConverged()) return m_pParam.get();				// we have converged

			m_paramID = (m_paramID + 1) % m_nParams;		// new argument

			// reset variabels for new argument
			m_vKappa[mD] = -1;
			m_vKappa[pD] = -1;
			m_nSteps = 0;
			m_koeff = 1.0;

			m_midPoint = curArg;							// refresh the middle point
		}
		else if (maxKappa == m_vKappa[mD]) {	// >>>>> Lower value -> Step argument down
			std::fill(m_vConverged.begin(), m_vConverged.end(), false);		// reset convergence

			m_midPoint = MAX(minArg, m_midPoint - m_koeff * delta);		// refresh the middle point

			// shift kappa
			m_vKappa[pD] = m_vKappa[oD];
			m_vKappa[oD] = m_vKappa[mD];
			m_vKappa[mD] = -1.0f;

			// increase the search step
			m_nSteps++;
			m_koeff += m_acceleration * m_nSteps;
		}
		else if (maxKappa == m_vKappa[pD]) {	// >>>>> Upper value -> Step argument up
			std::fill(m_vConverged.begin(), m_vConverged.end(), false);		// reset convergence

			m_midPoint = MIN(maxArg, m_midPoint + m_koeff * delta);		// refresh the middle point

			// shift kappa
			m_vKappa[mD] = m_vKappa[oD];
			m_vKappa[oD] = m_vKappa[pD];
			m_vKappa[pD] = -1.0f;

			// increase the search step
			m_nSteps++;
			m_koeff += m_acceleration * m_nSteps;
		}
	} // infinite loop
}

bool CPowell::isConverged(void)
{
	for (auto &converged : m_vConverged) if (!converged) return false;
	return true;
}


}