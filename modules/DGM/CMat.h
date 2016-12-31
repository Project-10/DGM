// Confusion matrix class	
// Written by Sergey G. Kosov in 2013 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	class CPriorEdge;

	// ================================ Confusion Matrix Class ================================
	/**
	* @ingroup moduleEva
	* @brief Confusion matrix class.
	* @details This class estimates the <a href="http://en.wikipedia.org/wiki/Confusion_matrix" target="_blank">confusion matrix</a>, which  
	* allows for evaluation of the classification results in terms of the overall classification accuracy, as well as of
	* <a href="http://en.wikipedia.org/wiki/Precision_and_recall" target="_blank">precision and recall</a>. 
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/	
	class CCMat
	{
	public:
		/**
		* @brief Constructor
		* @param nStates Number of states (classes)
		*/
		DllExport CCMat(byte nStates);
		DllExport virtual ~CCMat(void);

		/**
		* @brief Resets class variables.
		* @details Allows to re-use the class.
		*/		
		DllExport void	reset(void);
		/**
		* @brief Saves the confusion matrix.
		* @details Allows to re-use the class. Stores data to the file: \b "<path><name>_<idx>.dat". 
		* @param path Path to the destination folder. 
		* @param name Name of data file. If empty, will be generated automatically from the class name.
		* @param idx Index of the destination file. Negative value means no index.
		*/		
		DllExport void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const;
		/**
		* @brief Loads the confusion matrix.
		* @details Allows to re-use the class. Loads data to the file: \b "<path><name>_<idx>.dat". 
		* @param path Path to the folder, containing the data file.
		* @param name Name of data file. If empty, will be generated automatically from the class name.
		* @param idx Index of the data file. Negative value means no index.
		*/		
		DllExport void	load(const std::string &path, const std::string &name = std::string(), short idx = -1);
		/**
		* @brief Estimates the confusion matrix
		* @param gt Matrix, each element of which is a ground-truth state (class)
		* @param solution Matrix with the predicted states, provided by classifier
		* @param mask Operation mask. Its non-zero elements indicate which matrix elements need to be stimated. 
		* Mask should have the same size as \b gt and \b solution and be of type \b CV_8UC1 .
		*/
		DllExport void  estimate(const Mat &gt, const Mat &solution, const Mat &mask = Mat());
		/**
		* @brief Estimates the confusion matrix
		* @param gt	The ground-truth state (class)
		* @param solution The predicted state, provided by classifier
		*/
		DllExport void	estimate(byte gt, byte solution);
		
		/**
		* @brief Returns the confusion matrix
		* @details The resulting confusion matrix may be visualized with help of CMarker::drawConfusionMatrix() function
		* @returns Confusion matrix, where each value is given in percents. Mat(size: nStates x nStates; type: CV_32FC1)
		*/
		DllExport Mat	getConfusionMatrix(void) const;
		/**
		* @brief Returns the overall classification accuracy
		* @returns The classification accuracy: percent of the correctly classified samples.
		*/
		DllExport float	getAccuracy(void) const;	

//		Old version
//		DllExport void	print(char *fileName, int shiftBase);
//		DllExport void	getAccuracy(int shiftBase, double *base, double *occlusion, double *overall);


	protected:
		CPriorEdge	* m_pConfusionMatrix;		///< COnfusion matrix container

	//	DllExport void	saveFile(FILE *pFile) const {CPriorEdge::saveFile(pFile);} 
	//	DllExport void	loadFile(FILE *pFile) {CPriorEdge::loadFile(pFile);}		


	};
}
