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

#include "types.h"

namespace DirectGraphicalModels 
{
	#define CV_DTREE_CAT_DIR(idx,subset) (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)	
	
	template<typename _Tp> struct cmp_lt_idx
	{
		cmp_lt_idx(const _Tp* _arr) : arr(_arr) {}
		bool operator ()(int a, int b) const { return arr[a] < arr[b]; }
		const _Tp* arr;
	};

	template<typename _Tp> struct cmp_lt_ptr
	{
		cmp_lt_ptr() {}
		bool operator ()(const _Tp* a, const _Tp* b) const { return *a < *b; }
	};

	static inline void setRangeVector(vec_int_t& vec, int n)
	{
		vec.resize(n);
		for (int i = 0; i < n; i++)
			vec[i] = i;
	}

	struct TreeParams
	{
	public:
		bool  useSurrogates;
		bool  use1SERule;
		bool  truncatePrunedTree;
		Mat	  priors;

	protected:
		int   maxCategories;
		int   maxDepth;
		int   minSampleCount;
		int   CVFolds;
		float regressionAccuracy;

	public:
		TreeParams() : useSurrogates(false), use1SERule(true), truncatePrunedTree(true), priors(Mat()), 
					   maxCategories(10), maxDepth(INT_MAX), minSampleCount(10), CVFolds(10), regressionAccuracy(0.01f) {}
		TreeParams(int _maxDepth, int _minSampleCount, float _regressionAccuracy, bool _useSurrogates,	int _maxCategories, int _CVFolds, bool _use1SERule, bool _truncatePrunedTree, const Mat& _priors)
					 : useSurrogates(_useSurrogates), use1SERule(_use1SERule), truncatePrunedTree(_truncatePrunedTree), priors(_priors),
					   maxCategories(_maxCategories), maxDepth(_maxDepth), minSampleCount(_minSampleCount), CVFolds(_CVFolds), regressionAccuracy(_regressionAccuracy) {}

		inline void  setMaxCategories(int val) {
			if (val < 2)
				CV_Error(CV_StsOutOfRange, "max_categories should be >= 2");
			maxCategories = std::min(val, 15);
		}
		inline void  setMaxDepth(int val) {
			if (val < 0)
				CV_Error(CV_StsOutOfRange, "max_depth should be >= 0");
			maxDepth = std::min(val, 25);
		}
		inline void  setMinSampleCount(int val) { minSampleCount = std::max(val, 1); }
		inline void  setCVFolds(int val) {
			if (val < 0)
				CV_Error(CV_StsOutOfRange,
					"params.CVFolds should be =0 (the tree is not pruned) "
					"or n>0 (tree is pruned using n-fold cross-validation)");
			if (val == 1)
				val = 0;
			CVFolds = val;
		}
		inline void  setRegressionAccuracy(float val) {
			if (val < 0)
				CV_Error(CV_StsOutOfRange, "params.regression_accuracy should be >= 0");
			regressionAccuracy = val;
		}

		inline int	 getMaxCategories() const { return maxCategories; }
		inline int	 getMaxDepth() const { return maxDepth; }
		inline int	 getMinSampleCount() const { return minSampleCount; }
		inline int	 getCVFolds() const { return CVFolds; }
		inline float getRegressionAccuracy() const { return regressionAccuracy; }
	};
	
	/** 
	* @brief The class represents a single decision tree or a collection of decision trees.
	* @details The current public interface of the class allows user to train only a single decision tree, however
	* the class is capable of storing multiple decision trees and using them for prediction (by summing
	* responses or using a voting schemes), and the derived from CDTrees classes (such as CRTrees and CBoost)
	* use this capability to implement decision tree ensembles.
	*/
	class CDTrees : public ml::StatModel
	{
	public:
		/// Predict options 
		enum Flags { PREDICT_AUTO = 0, PREDICT_SUM = (1 << 8), PREDICT_MAX_VOTE = (2 << 8), PREDICT_MASK = (3 << 8) };

		/** 
		* @brief Creates the empty model
		* The static method creates empty decision tree with the specified parameters. It should be then
		* trained using train method (see StatModel::train). Alternatively, you can load the model from
		* file using Algorithm::load\<CDTrees\>(filename).
		*/
		static Ptr<CDTrees> create() { return makePtr<CDTrees>(); }
		
		CDTrees(void) : ml::StatModel() {}
		virtual ~CDTrees(void) {}
		
		/** 
		* @brief Cluster possible values of a categorical variable into K\<=maxCategories clusters to find a suboptimal split.
		* @details If a discrete variable, on which the training procedure tries to make a split, takes more than
		* maxCategories values, the precise best subset estimation may take a very long time because the
		* algorithm is exponential. Instead, many decision trees engines (including our implementation)
		* try to find sub-optimal split in this case by clustering all the samples into maxCategories
		* clusters that is some categories are merged together. The clustering is applied only in n \>
		* 2-class classification problems for categorical variables with N \> max_categories possible
		* values. In case of regression and 2-class classification the optimal split can be found
		* efficiently without employing clustering, thus the parameter is not used in these cases.
		* Default value is 10.
		* @see setMaxCategories 
		*/
		virtual int							getMaxCategories(void) const { return params.getMaxCategories(); }
		/** 
		* @copybrief getMaxCategories @see getMaxCategories 
		*/
		virtual void						setMaxCategories(int val) { params.setMaxCategories(val); }
		/** 
		* @brief The maximum possible depth of the tree.
		* @details That is the training algorithms attempts to split a node while its depth is less than maxDepth.
		* The root node has zero depth. The actual depth may be smaller if the other termination criteria
		* are met, and/or if the tree is pruned. Default value is INT_MAX.
		* @see setMaxDepth */
		virtual int							getMaxDepth(void) const { return params.getMaxDepth(); }
		/** 
		* @copybrief getMaxDepth @see getMaxDepth 
		*/
		virtual void						setMaxDepth(int val) { params.setMaxDepth(val); }
		/** 
		* @brief If the number of samples in a node is less than this parameter then the node will not be split.
		* @details Default value is 10.
		* @see setMinSampleCount 
		*/
		virtual int							getMinSampleCount(void) const { return params.getMinSampleCount(); }
		/** 
		* @copybrief getMinSampleCount @see getMinSampleCount 
		*/
		virtual void						setMinSampleCount(int val) { params.setMinSampleCount(val); }
		/** 
		* @brief If CVFolds \> 1 then algorithms prunes the built decision tree using K-fold
		* cross-validation procedure where K is equal to CVFolds.
		* @details Default value is 10.
		* @see setCVFolds 
		*/
		virtual int							getCVFolds(void) const { return params.getCVFolds(); }
		/** 
		* @copybrief getCVFolds @see getCVFolds 
		*/
		virtual void						setCVFolds(int val) { params.setCVFolds(val); }
		/** 
		* @brief If true then surrogate splits will be built.
		* @details These splits allow to work with missing data and compute variable importance correctly.
		* Default value is false.
		* @note currently it's not implemented.
		* @see setUseSurrogates 
		*/
		virtual bool						getUseSurrogates(void) const { return params.useSurrogates; }
		/** 
		* @copybrief getUseSurrogates @see getUseSurrogates 
		*/
		virtual void						setUseSurrogates(bool val) { params.useSurrogates = val; }
		/** 
		* @brief If true then a pruning will be harsher.
		* @details This will make a tree more compact and more resistant to the training data noise but a bit less
		* accurate. Default value is true.
		* @see setUse1SERule 
		*/
		virtual bool						getUse1SERule(void) const { return params.use1SERule; }
		/** 
		* @copybrief getUse1SERule @see getUse1SERule 
		*/
		virtual void						setUse1SERule(bool val) { params.use1SERule = val; }
		/** 
		* @brief If true then pruned branches are physically removed from the tree.
		* @details Otherwise they are retained and it is possible to get results from the original unpruned (or
		pruned less aggressively) tree. Default value is true.
		* @see setTruncatePrunedTree 
		*/
		virtual bool						getTruncatePrunedTree(void) { return params.truncatePrunedTree; }
		/** 
		* @copybrief getTruncatePrunedTree @see getTruncatePrunedTree 
		*/
		virtual void						setTruncatePrunedTree(bool val) { params.truncatePrunedTree = val; }
		/** 
		* @brief Termination criteria for regression trees.
		* If all absolute differences between an estimated value in a node and values of train samples
		* in this node are less than this parameter then the node will not be split further. Default
		* value is 0.01f
		* @see setRegressionAccuracy 
		*/
		virtual float						getRegressionAccuracy(void) const { return params.getRegressionAccuracy(); };
		/** 
		* @copybrief getRegressionAccuracy @see getRegressionAccuracy 
		*/
		virtual void						setRegressionAccuracy(float val) { params.setRegressionAccuracy(val); }
		/** 
		* @brief The array of a priori class probabilities, sorted by the class label value.
		* @details The parameter can be used to tune the decision tree preferences toward a certain class. For
		* example, if you want to detect some rare anomaly occurrence, the training base will likely
		* contain much more normal cases than anomalies, so a very good classification performance
		* will be achieved just by considering every case as normal. To avoid this, the priors can be
		* specified, where the anomaly probability is artificially increased (up to 0.5 or even
		* greater), so the weight of the misclassified anomalies becomes much bigger, and the tree is
		* adjusted properly.
		*
		* You can also think about this parameter as weights of prediction categories which determine
		* relative weights that you give to misclassification. That is, if the weight of the first
		* category is 1 and the weight of the second category is 10, then each mistake in predicting
		* the second category is equivalent to making 10 mistakes in predicting the first category.
		* Default value is empty Mat.
		* @see setPriors 
		*/
		virtual Mat							getPriors(void) const { return params.priors; }
		/** 
		* @copybrief getPriors @see getPriors 
		*/
		virtual void						setPriors(const Mat &val) { params.priors = val; }		

		/** 
		* @brief The class represents a decision tree node.
		*/
		class Node
		{
		public:
			Node();
			double	value;		///< Value at the node: a class label in case of classification or estimated function value in case of regression.
			int		classIdx;	///< Class index normalized to 0..class_count-1 range and assigned to the node. It is used internally in classification trees and tree ensembles.
			int		parent;		///< Index of the parent node
			int		left;		///< Index of the left child node
			int		right;		///< Index of right child node
			int		defaultDir; ///< Default direction where to go (-1: left or +1: right). It helps in the case of missing values.
			int		split;		///< Index of the first split
		};

		/** 
		* @brief The class represents split in a decision tree.
		*/
		class Split
		{
		public:
			Split();
			int		varIdx;		///< Index of variable on which the split is created.
			bool	inversed;	///< If true, then the inverse split rule is used (i.e. left and right branches are exchanged in the rule expressions below).
			float	quality;	///< The split quality, a positive number. It is used to choose the best split.
			int		next;		///< Index of the next split in the list of splits for the node
			float	c;			/**< The threshold value in case of split on an ordered variable.
										The rule is:
										@code{.none}
										if var_value < c
										then next_node <- left
										else next_node <- right
										@endcode */
			int	subsetOfs;		/**< Offset of the bitset used by the split on a categorical variable.
										The rule is:
										@code{.none}
										if bitset[var_value] == 1
										then next_node <- left
										else next_node <- right
										@endcode */
		};

		struct WNode
		{
			int		class_idx;
			double	Tn;
			double	value;

			int		parent;
			int		left;
			int		right;
			int		defaultDir;

			int		split;

			int		sample_count;
			int		depth;
			double	maxlr;

			// global pruning data
			int		complexity;
			double	alpha;
			double	node_risk, tree_risk, tree_error;

			WNode() {
				class_idx = sample_count = depth = complexity = 0;
				parent = left = right = split = defaultDir = -1;
				Tn = INT_MAX;
				value = maxlr = alpha = node_risk = tree_risk = tree_error = 0.;
			}
		};

		struct WSplit
		{
			int		varIdx;
			bool	inversed;
			float	quality;
			int		next;
			float	c;
			int		subsetOfs;

			WSplit() {
				varIdx = next = 0;
				inversed = false;
				quality = c = 0.f;
				subsetOfs = -1;
			}
		};

		struct WorkData
		{
			Ptr<ml::TrainData>	data;
			std::vector<WNode>	wnodes;
			std::vector<WSplit>	wsplits;
			vec_int_t			wsubsets;
			std::vector<double> cv_Tn;
			std::vector<double> cv_node_risk;
			std::vector<double> cv_node_error;
			vec_int_t			cv_labels;
			std::vector<double> sample_weights;
			vec_int_t			cat_responses;
			std::vector<double> ord_responses;
			vec_int_t			sidx;
			int					maxSubsetSize;

			WorkData(const Ptr<ml::TrainData>& _data) {
				data = _data;
				vec_int_t subsampleIdx;
				Mat sidx0 = _data->getTrainSampleIdx();
				if (!sidx0.empty()) {
					sidx0.copyTo(sidx);
					std::sort(sidx.begin(), sidx.end());
				} else {
					int n = _data->getNSamples();
					setRangeVector(sidx, n);
				}
				maxSubsetSize = 0;
			}
		};
		
		virtual void						clear(void);

		String								getDefaultName(void) const { return "opencv_ml_dtree"; }
		bool								isTrained(void) const { return !roots.empty(); }
		bool								isClassifier(void) const { return _isClassifier; }
		int									getVarCount(void) const { return varType.empty() ? 0 : (int)(varType.size() - 1); }
		int									getCatCount(int vi) const { return catOfs[vi][1] - catOfs[vi][0]; }
		int									getSubsetSize(int vi) const { return (getCatCount(vi) + 31) / 32; }

		virtual void						setDParams(const TreeParams& _params);
		virtual void						startTraining(const Ptr<ml::TrainData>& trainData, int flags);
		virtual void						endTraining(void);
		virtual void						initCompVarIdx(void);
		virtual bool						train(const Ptr<ml::TrainData>& trainData, int flags = 0);

		virtual int							addTree(const vec_int_t& sidx);
		virtual int							addNodeAndTrySplit(int parent, const vec_int_t& sidx);
		virtual const vec_int_t			  & getActiveVars(void);
		virtual int							findBestSplit(const vec_int_t& _sidx);
		virtual void						calcValue(int nidx, const vec_int_t& _sidx);

		virtual WSplit						findSplitOrdClass(int vi, const vec_int_t& _sidx, double initQuality);

		// simple k-means, slightly modified to take into account the "weight" (L1-norm) of each vector.
		virtual void						clusterCategories(const double* vectors, int n, int m, double* csums, int k, int* labels);
		virtual WSplit						findSplitCatClass(int vi, const vec_int_t& _sidx, double initQuality, int* subset);

		virtual WSplit						findSplitOrdReg(int vi, const vec_int_t& _sidx, double initQuality);
		virtual WSplit						findSplitCatReg(int vi, const vec_int_t& _sidx, double initQuality, int* subset);

		virtual int							calcDir(int splitidx, const vec_int_t& _sidx, vec_int_t& _sleft, vec_int_t& _sright);
		virtual int							pruneCV(int root);

		virtual double						updateTreeRNC(int root, double T, int fold);
		virtual bool						cutTree(int root, double T, int fold, double min_alpha);
		virtual float						predictTrees(const Range& range, const Mat& sample, int flags) const;
		virtual float						predict(InputArray inputs, OutputArray outputs = noArray(), int flags = 0) const;

		virtual void						writeTrainingParams(FileStorage& fs) const;
		virtual void						writeParams(FileStorage& fs) const;
		virtual void						writeSplit(FileStorage& fs, int splitidx) const;
		virtual void						writeNode(FileStorage& fs, int nidx, int depth) const;
		virtual void						writeTree(FileStorage& fs, int root) const;
		virtual void						write(FileStorage& fs) const;

		virtual void						readParams(const FileNode& fn);
		virtual int							readSplit(const FileNode& fn);
		virtual int							readNode(const FileNode& fn);
		virtual int							readTree(const FileNode& fn);
		virtual void						read(const FileNode& fn);

		/// @brief Returns indices of root nodes
		virtual const vec_int_t			  & getRoots(void) const { return roots; }
		/// @brief Returns all the nodes all the node indices are indices in the returned vector
		virtual const std::vector<Node>	  & getNodes(void) const { return nodes; }
		/// @brief Returns all the splits all the split indices are indices in the returned vector
		virtual const std::vector<Split>  & getSplits(void) const { return splits; }
		/// @brief Returns all the bitsets for categorical splits Split::subsetOfs is an offset in the returned vector
		virtual const vec_int_t			  & getSubsets(void) const { return subsets; }


protected:		
		TreeParams							params;
		vec_int_t							varIdx;
		vec_int_t							compVarIdx;
		std::vector<uchar>					varType;
		std::vector<Vec2i>					catOfs;
		vec_int_t							catMap;
		vec_int_t							roots;
		std::vector<Node>					nodes;
		std::vector<Split>					splits;
		vec_int_t							subsets;
		vec_int_t							classLabels;
		vec_float_t							missingSubst;
		vec_int_t							varMapping;
		bool								_isClassifier;

		Ptr<WorkData>						w;
	};

	template <typename T> static inline void readVectorOrMat(const FileNode & node, std::vector<T> & v)
	{
		if (node.type() == FileNode::MAP)
		{
			Mat m;
			node >> m;
			m.copyTo(v);
		}
		else if (node.type() == FileNode::SEQ)
		{
			node >> v;
		}
	}
}
