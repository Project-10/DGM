/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#pragma once

#include "DTrees.h"

namespace DirectGraphicalModels
{
	struct RTreeParams
	{
		bool		 calcVarImportance;
		int			 nactiveVars;
		TermCriteria termCrit;

		RTreeParams() : calcVarImportance(false), nactiveVars(0), termCrit(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 50, 0.1)) {}
		RTreeParams(bool _calcVarImportance, int _nactiveVars, TermCriteria _termCrit) : calcVarImportance(_calcVarImportance), nactiveVars(_nactiveVars), termCrit(_termCrit) {}
	};
	
	/** 
	* @brief The class implements the random forest predictor.
	*/
	class CRTrees : public CDTrees
	{
	public:
		/** 
		* @brief Creates the empty model.
		* @details Use StatModel::train to train the model, StatModel::train to create and train the model,
		* Algorithm::load to load the pre-trained model.
		*/
		static Ptr<CRTrees>		create(void) { return makePtr<CRTrees>(); }

		CRTrees(void);
		virtual ~CRTrees(void);

		void					clear(void);
		const vec_int_t		  & getActiveVars(void);
		void					startTraining(const Ptr<ml::TrainData> &trainData, int flags);
		void					endTraining(void);
		bool					train(const Ptr<ml::TrainData> &trainData, int flags = 0);
		void					writeTrainingParams(FileStorage& fs) const;
		void					write(FileStorage &fs) const;
		void					readParams(const FileNode &fn);
		void					read(const FileNode &fn);

		/** 
		* @brief If true then variable importance will be calculated and then it can be retrieved by 
		* CRTrees::getVarImportance. 
		* @details Default value is false.
		* @see setCalculateVarImportance 
		*/
		virtual bool			getCalculateVarImportance(void) const { return rparams.calcVarImportance; }
		/**
		* @copybrief getCalculateVarImportance 
		* @see getCalculateVarImportance 
		*/
		virtual void			setCalculateVarImportance(bool val) { rparams.calcVarImportance = val; }
		/** 
		* @brief The size of the randomly selected subset of features at each tree node and that are used
		* to find the best split(s).
		* @details If you set it to 0 then the size will be set to the square root of the total number of
		* features. Default value is 0.
		* @see setActiveVarCount */
		virtual int				getActiveVarCount(void) const { return rparams.nactiveVars; }
		/** 
		* @copybrief getActiveVarCount @see getActiveVarCount 
		*/
		virtual void			setActiveVarCount(int val) { rparams.nactiveVars = val; }
		/** 
		* @brief The termination criteria that specifies when the training algorithm stops.
		* @details Either when the specified number of trees is trained and added to the ensemble or when
		* sufficient accuracy (measured as OOB error) is achieved. Typically the more trees you have the
		* better the accuracy. However, the improvement in accuracy generally diminishes and asymptotes
		* pass a certain number of trees. Also to keep in mind, the number of tree increases the
		* prediction time linearly. Default value is TermCriteria(TermCriteria::MAX_ITERS +
		*  TermCriteria::EPS, 50, 0.1)
		* @see setTermCriteria */
		virtual TermCriteria	getTermCriteria(void) const { return rparams.termCrit; }
		/** 
		* @copybrief getTermCriteria @see getTermCriteria 
		*/
		virtual void			setTermCriteria(const TermCriteria &val) { rparams.termCrit = val; }
		/** 
		* @brief Returns the variable importance array.
		* @details The method returns the variable importance vector, computed at the training stage when
		* CalculateVarImportance is set to true. If this flag was set to false, the empty matrix is
		* returned.
		*/
		virtual Mat				getVarImportance(void) const { return Mat_<float>(varImportance, true); }


		
	protected:		
		RTreeParams rparams;
		double		oobError;
		vec_float_t	varImportance;
		vec_int_t	allVars;
		vec_int_t	activeVars;
		RNG			rng;
	};
}
