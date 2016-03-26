// Base abstract class for random model training
// Written by Sergey G. Kosov in 2015 for Project X
#pragma once

#include "types.h"

namespace DirectGraphicalModels
{
	/**
	* @brief Random model types.
	* @details Define the maximal number of nodes in the cliques.
	*/
	enum RandomModelType {
		RM_UNARY = 1,	///< Unary random model: no iteraction between nodes.
		RM_PAIRWISE,	///< Pairwise random model: maximum two nodes in the cliques.
		RM_TRIPLET		///< %Triplet random model: maximum tree nodes in the cliques.
	};
// ================================ Base Random Model Class ================================
	/**
	* @brief Base abstract class for random model training.
	* @details This class defines basic serialization interface.
	* @author Sergey G. Kosov, sergey.kosov@project-10.de
	*/
	class CBaseRandomModel
	{
	public:
		/**
		* @brief Constructor.
		* @param nStates Number of states (classes).
		*/
		CBaseRandomModel(byte nStates) : m_nStates(nStates) {}
		virtual ~CBaseRandomModel(void) {}

		/**
		* @brief Resets class variables.
		* @details Allows to re-use the class.
		*/
		DllExport virtual void	reset(void) = 0;	
		/**
		* @brief Saves the training data.
		* @details Allows to re-use the class. Stores data to the file: \b "<path><name>_<idx>.dat". 
		* @param path Path to the destination folder. 
		* @param name Name of data file. If empty, will be generated automatically from the class name.
		* @param idx Index of the destination file. Negative value means no index.
		*/
		DllExport virtual void	save(const std::string &path, const std::string &name = std::string(), short idx = -1) const;
		/**
		* @brief Loads the training data.
		* @details Allows to re-use the class. Loads data to the file: \b "<path><name>_<idx>.dat". 
		* @param path Path to the folder, containing the data file.
		* @param name Name of data file. If empty, will be generated automatically from the class name.
		* @param idx Index of the data file. Negative value means no index.
		*/		
		DllExport virtual void	load(const std::string &path, const std::string &name = std::string(), short idx = -1); 
		/**
		* @brief Returns number of features
		* @return Number of features 
		*/		
		DllExport byte			getNumStates(void) const {return m_nStates;}	


	protected:
		/**
		* @brief Saves the random model into the file.
		* @details Allows to re-use the class. 
		* @param pFile Pointer to the file, opened for writing.
		*/
		DllExport virtual void	saveFile(FILE *pFile) const = 0;
		/**
		* @brief Loads the random model from the file.
		* @details Allows to re-use the class.
		* @param pFile Pointer to the file, opened for reading.
		*/	
		DllExport virtual void	loadFile(FILE *pFile) = 0;
		/**
		* @brief Generates name of the data file for storing random model parameters.
		* @details This function generated the file name as follows: \b fileName="<path><name>_<idx>.dat", where \b idx always has 5 symbols. 
		* @param path Path to the folder, containing the data file.
		* @param name Name of data file.
		* @param idx Index of the data file. If idx is negative, index will not be added to the file name.		
		* @returns File name.
		*/
		inline std::string generateFileName(const std::string &path, const std::string &name, short idx) const;


	protected:
		byte	m_nStates;		///< The number of states (classes)

	};
}