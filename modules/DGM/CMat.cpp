#include "CMat.h"
#include "PriorEdge.h"
#include "macroses.h"

namespace DirectGraphicalModels
{
// Constructor
CCMat::CCMat(byte nStates)
{
	m_pConfusionMatrix = new CPriorEdge(nStates, eP_APP_NORM_STANDARD);
}

// Destructor
CCMat::~CCMat(void)
{
	delete m_pConfusionMatrix;
}

void CCMat::reset(void) 
{
	m_pConfusionMatrix->reset(); 
}

void CCMat::save(const std::string &path, const std::string &name, short idx) const 
{
	m_pConfusionMatrix->save(path, name.empty() ? "CMat" : name, idx); 
}

void CCMat::load(const std::string &path, const std::string &name, short idx) 
{
	m_pConfusionMatrix->load(path, name.empty() ? "CMat" : name, idx); 
} 

void CCMat::estimate(const Mat &gt, const Mat &solution) 
{ 
	DGM_ELEMENTWISE2<CCMat, &CCMat::estimate>(*this, gt, solution); 
}

void CCMat::estimate(const Mat &gt, const Mat &solution, const Mat &mask)
{
	DGM_ELEMENTWISE2<CCMat, &CCMat::estimate>(*this, gt, solution, mask);
}

void CCMat::estimate(byte gt, byte solution) 
{ 
	m_pConfusionMatrix->addEdgeGroundTruth(solution, gt); 
}

Mat CCMat::getConfusionMatrix(void) const 
{
	return m_pConfusionMatrix->getPrior(100.0f); 
}

float CCMat::getAccuracy(void) const
{
	float res	= 0;
	Mat   cMat	= getConfusionMatrix();
	for (byte s = 0; s < m_pConfusionMatrix->getNumStates(); s++)
		res += cMat.at<float>(s, s);
	return res;
}

// Old version
/*void CCMat::print(char *fileName, int shiftBase)
{
	FILE	*pFile;
	float	 val;
	long	 acc  = 0;			// average accuracy
	long	 accb = 0;			// base level accuracy
	long	 acco = 0;			// occlusion level accuaray
	long	 n	= 0;			// number of all counted samples
	long	 nb = 0;			// number of counted samples for the base layer
	long	 no = 0;			// number of counted samples for the occlusion layer
	int		 nX, nY;
	char	 str[8];
	
	for (nX = 0; nX < m_nStates; nX++) 
		for (nY = 0; nY < m_nStates; nY++) {
			if ((nX < shiftBase) && (nY < shiftBase)) nb += m_pCMat[nX * m_nStates + nY];
			else no += m_pCMat[nX * m_nStates + nY];
		} // nY
	n = nb + no;

	pFile = fopen(fileName, "w+");
	fprintf(pFile, "Confusion Matrix:\n");
	fprintf(pFile, "-----------------\n");
	fprintf(pFile, "R\\C\t");
	for (nX = 0; nX < m_nStates; nX++) fprintf(pFile, "%d\t", nX); fprintf(pFile, "Comp\n");
	for (nY = 0; nY < m_nStates; nY++) {
		fprintf(pFile, "%d\t", nY);
		if (nY < shiftBase) accb += m_pCMat[nY * m_nStates + nY];
		else acco += m_pCMat[nY * m_nStates + nY];
		long comp = 0;
		for (nX = 0; nX < m_nStates; nX++) {
			comp += m_pCMat[nX * m_nStates + nY];											// Completness
			if (n == 0)	val = 0;  
			else		val = 100 * static_cast<float>(m_pCMat[nX * m_nStates + nY]) / n;
			if (val == 0) sprintf(str, "--.--"); else
			if (val < 10) sprintf(str, " %.2f%%", val); else
			sprintf(str, "%.2f%%", val);
			fprintf(pFile, "%s\t", str);
		} // nY
		if (comp == 0)	val = 0; 
		else			val = 100 * static_cast<float>(m_pCMat[nY * m_nStates + nY]) / comp;
		if (val == 0) sprintf(str, "--.--"); else
		if (val < 10) sprintf(str, " %.2f%%", val); else
		sprintf(str, "%.2f%%", val);
		fprintf(pFile, "%s\t", str);

		fprintf(pFile, "\n");
	} // nX
	fprintf(pFile, "Corr\t");
	for (nX = 0; nX < m_nStates; nX++) {
		long corr = 0;
		for (nY = 0; nY < m_nStates; nY++) 
			corr += m_pCMat[nX * m_nStates + nY]; 
		if (corr == 0)	val = 0; 
		else			val = 100 * static_cast<float>(m_pCMat[nX * m_nStates + nX]) / corr;
		if (val == 0) sprintf(str, "--.--"); else
		if (val < 10) sprintf(str, " %.2f%%", val); else
		sprintf(str, "%.2f%%", val);
		fprintf(pFile, "%s\t", str);
	} // nX
	fprintf(pFile, "\n");
	
	acc = accb + acco;

	fprintf(pFile, "\nOverall accuracy:\n");
	fprintf(pFile, "-----------------\n");
	fprintf(pFile, "%.2f%% (base: %.2f%%) (occl: %.2f%%)\n", 100 * static_cast<double>(acc) / n, 100 * static_cast<double>(accb) / nb, 100 * static_cast<double>(acco) / no);


	printf("\nOverall accuracy:\n");
	printf("-----------------\n");
	printf("%.2f%% (base: %.2f%%) (occl: %.2f%%)\n", 100 * static_cast<double>(acc) / n, 100 * static_cast<double>(accb) / nb, 100 * static_cast<double>(acco) / no);

	fclose(pFile);	
}

void CCMat::getAccuracy(int shiftBase, double *base, double *occlusion, double *overall)
{
	long	 accb = 0;			// base level accuracy
	long	 acco = 0;			// occlusion level accuaray
	long	 nb = 0;			// number of counted samples for the base layer
	long	 no = 0;			// number of counted samples for the occlusion layer
	int		 nX, nY;
	
	for (nX = 0; nX < m_nStates; nX++) 
		for (nY = 0; nY < m_nStates; nY++) {
			if ((nX < shiftBase) && (nY < shiftBase)) nb += m_pCMat[nX * m_nStates + nY];
			else no += m_pCMat[nX * m_nStates + nY];
		} // nY

	for (nY = 0; nY < m_nStates; nY++) {
		if (nY < shiftBase) accb += m_pCMat[nY * m_nStates + nY];
		else acco += m_pCMat[nY * m_nStates + nY];
	} // nX

	if (nb != 0) *base		= static_cast<double>(accb) / nb; else *base		= 0;
	if (no != 0) *occlusion	= static_cast<double>(acco) / no; else *occlusion	= 0;
	*overall = (*base + *occlusion) / 2;

}
*/
}