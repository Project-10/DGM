#pragma once

// This file defines the ParallelForestTrainer and ParallelTreeTraininer classes,
// which are responsible for creating new Tree instances by learning from
// training data. These classes have almost identical interfaces to ForestTrainer
// and TreeTrainer, but allow candidate feature evaluation to be shared over a
// specified maximum number of threads.

// *** NOTE *** Compiling this header requires OpenMP.

#include <assert.h>

#include <vector>
#include <string>
#include <algorithm>

#include <omp.h>

#include "ProgressStream.h"

#include "TrainingParameters.h"
#include "Interfaces.h"
#include "Tree.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  class Random;
  
  /// <summary>
  /// A decision tree training operation in which candidate feature response
  /// function evaluation is distributed over multiple threads - used
  /// internally within ParallelTreeTrainer to encapsulate the training a single tree.
  /// </summary>
  template<class F, class S>
  class ParallelTreeTrainingOperation // where F : IFeatureResponse where S : IStatisticsAggregator<S>
  {
  private:
    typedef typename std::vector<Node<F,S> >::size_type NodeIndex;
    typedef typename std::vector<unsigned int>::size_type DataPointIndex;

    Random& random_;

    const IDataPointCollection& data_;

    ITrainingContext<F, S>& trainingContext_;

    TrainingParameters parameters_;

    int maxThreads_;

    S parentStatistics_, leftChildStatistics_, rightChildStatistics_;

    std::vector<S> partitionStatistics_;
    std::vector<float> responses_;
    std::vector<unsigned int> indices_;

    ProgressStream progress_;

    class ThreadLocalData
    {
    public:
      double maxGain;
      F bestFeature;
      float bestThreshold;

      S parentStatistics_, leftChildStatistics_, rightChildStatistics_;

      std::vector<S> partitionStatistics_;
      std::vector<float> responses_;
      std::vector<float> thresholds;

      Random random_;

      ThreadLocalData()
      {

      }

      ThreadLocalData(Random& random, ITrainingContext<F,S>& trainingContext_, const TrainingParameters& parameters, IDataPointCollection const & data):random_(random.Next())
      {
        maxGain = 0.0;
        bestThreshold = 0.0;
        parentStatistics_ = trainingContext_.GetStatisticsAggregator();

        leftChildStatistics_ = trainingContext_.GetStatisticsAggregator();
        rightChildStatistics_ = trainingContext_.GetStatisticsAggregator();

        partitionStatistics_.resize(parameters.NumberOfCandidateThresholdsPerFeature + 1);
        for (unsigned int i = 0; i < parameters.NumberOfCandidateThresholdsPerFeature + 1; i++)
          partitionStatistics_[i] = trainingContext_.GetStatisticsAggregator();

        responses_.resize(data.Count());
        // thresholds_ will be resized() in ChooseCandidateThresholds()
      }

      void Clear()
      {
        maxGain = 0.0;
        bestFeature = F();
        bestThreshold = 0.0f;
      }
    };

    std::vector<ThreadLocalData > threadLocalData_;

  public:
    ParallelTreeTrainingOperation(
      Random& random,
      ITrainingContext<F, S>& trainingContext,
      const TrainingParameters& parameters,
      int maxThreads,
      const IDataPointCollection& data,
      ProgressStream& progress):
    random_(random),
    data_(data),
    maxThreads_(maxThreads),
    trainingContext_(trainingContext),
    progress_(progress)
    {
      parameters_ = parameters;

      indices_ .resize(data.Count());
      for (DataPointIndex i = 0; i < indices_.size(); i++)
        indices_[i] = i;

      threadLocalData_.resize(maxThreads_);
      for (int threadIndex = 0; threadIndex < maxThreads_; threadIndex++)
        // Note use of placement new operator to initialize already-allocated memory
        new (&threadLocalData_[threadIndex]) ThreadLocalData(random, trainingContext_, parameters_, data_);
    }

    void TrainNodesRecurse(std::vector<Node<F, S> >& nodes, NodeIndex nodeIndex, DataPointIndex i0, DataPointIndex i1, int recurseDepth)
    {
      assert(nodeIndex < nodes.size());
      progress_[Verbose] << Tree<F, S>::GetPrettyPrintPrefix(nodeIndex) << i1 - i0 << ": ";

      // First aggregate statistics over the samples at the parent node
      parentStatistics_.Clear();
      for (DataPointIndex i = i0; i < i1; i++)
        parentStatistics_.Aggregate(data_, indices_[i]);

      // Copy parent statistics to thread local storage in case client IStatisticsAggregator implementations are not reentrant
      for (int t = 0; t < maxThreads_; t++)
        threadLocalData_[t].parentStatistics_ = parentStatistics_.DeepClone();

      if (nodeIndex >= nodes.size() / 2) // this is a leaf node, nothing else to do
      {
        nodes[nodeIndex].InitializeLeaf(parentStatistics_);
        progress_[Verbose] << "Terminating at max depth." << std::endl;
        return;
      }

      #pragma omp parallel for
      for(int threadIndex=0; threadIndex < maxThreads_; threadIndex++)
      {
        ThreadLocalData& tl = threadLocalData_[threadIndex]; // shorthand

        tl.Clear();

        // Iterate over candidate features
        std::vector<float> thresholds;
        for (int f = 0; f < parameters_.NumberOfCandidateFeatures/maxThreads_; f++)
        {
          F feature = trainingContext_.GetRandomFeature(tl.random_);

          for (unsigned int b = 0; b < parameters_.NumberOfCandidateThresholdsPerFeature + 1; b++)
            tl.partitionStatistics_[b].Clear(); // reset statistics

          // Compute feature response per samples at this node
          for (DataPointIndex i = i0; i < i1; i++)
            tl.responses_[i] = feature.GetResponse(data_, indices_[i]);

          int nThresholds;
          if ((nThresholds = ChooseCandidateThresholds(tl.random_, &indices_[0], i0, i1, &tl.responses_[0], tl.thresholds)) == 0)
            continue;

          // Aggregate statistics over sample partitions
          for (DataPointIndex i = i0; i < i1; i++)
          {
            int b = 0;
            while (b < nThresholds && tl.responses_[i] >= tl.thresholds[b])
              b++;

            tl.partitionStatistics_[b].Aggregate(data_, indices_[i]);
          }

          for (int t = 0; t < nThresholds; t++)
          {
            tl.leftChildStatistics_.Clear();
            tl.rightChildStatistics_.Clear();
            for (int p = 0; p < nThresholds + 1 /*i.e. nBins*/; p++)
            {
              if (p <= t)
                tl.leftChildStatistics_.Aggregate(tl.partitionStatistics_[p]);
              else
                tl.rightChildStatistics_.Aggregate(tl.partitionStatistics_[p]);
            }

            // Compute gain over sample partitions
            double gain = trainingContext_.ComputeInformationGain(tl.parentStatistics_, tl.leftChildStatistics_,tl. rightChildStatistics_);

            if (gain >= tl.maxGain)
            {
              tl.maxGain = gain;
              tl.bestFeature = feature;
              tl.bestThreshold = tl.thresholds[t];
            }
          }
        }
      }

      // Now merge over threads.
      double maxGain = 0.0;
      F bestFeature;
      float bestThreshold=0.0;

      for (int threadIndex = 0; threadIndex < maxThreads_; threadIndex++)
      {
        ThreadLocalData& tl = threadLocalData_[threadIndex];
        if (tl.maxGain > maxGain)
        {
          maxGain = tl.maxGain;
          bestFeature = tl.bestFeature;
          bestThreshold = tl.bestThreshold;
        }
      }

      if (maxGain == 0.0)
      {
        nodes[nodeIndex].InitializeLeaf(parentStatistics_);
        progress_[Verbose] << "Terminating with zero gain." << std::endl;
        return;
      }

      // Now reorder the data point indices using the winning feature and thresholds.
      // Also recompute child node statistics so the client can decide whether
      // to terminate training of this branch.
      leftChildStatistics_.Clear();
      rightChildStatistics_.Clear();

      for (DataPointIndex i = i0; i < i1; i++)
      {
        responses_[i] = bestFeature.GetResponse(data_, indices_[i]);
        if (responses_[i] < bestThreshold)
          leftChildStatistics_.Aggregate(data_, indices_[i]);
        else
          rightChildStatistics_.Aggregate(data_, indices_[i]);
      }

      if (trainingContext_.ShouldTerminate(parentStatistics_, leftChildStatistics_, rightChildStatistics_, maxGain))
      {
        nodes[nodeIndex].InitializeLeaf(parentStatistics_);
        progress_[Verbose] << "Terminating with no split." << std::endl;
        return;
      }

      // Otherwise this is a new decision node, recurse for children.
      nodes[nodeIndex].InitializeSplit(bestFeature, bestThreshold, parentStatistics_);

      // Now do partition sort - any sample with response greater goes left, otherwise right
      DataPointIndex ii = Tree<F, S>::Partition(responses_, indices_, i0, i1, bestThreshold);

      assert(ii >= i0 && i1 >= ii);

      progress_[Verbose] << " (threshold = " << bestThreshold << ", gain = "<< maxGain << ")." << std::endl;

      TrainNodesRecurse(nodes, nodeIndex * 2 + 1, i0, ii, recurseDepth + 1);
      TrainNodesRecurse(nodes, nodeIndex * 2 + 2, ii, i1, recurseDepth + 1);
    }

  private:
    int ChooseCandidateThresholds (
      Random& random,
      unsigned int* dataIndices, 
      DataPointIndex i0,
      DataPointIndex i1,
      const float* responses,
      std::vector<float>& thresholds )
    {
      thresholds.resize(parameters_.NumberOfCandidateThresholdsPerFeature + 1);
      std::vector<float>& quantiles = thresholds; // shorthand, for code clarity - we reuse memory to avoid allocation

      int nThresholds;
      // If there are enough response values...
      if (i1 - i0 > parameters_.NumberOfCandidateThresholdsPerFeature)
      {
        // ...make a random draw of NumberOfCandidateThresholdsPerFeature+1 response values
        nThresholds = parameters_.NumberOfCandidateThresholdsPerFeature;
        for (int i = 0; i < nThresholds + 1; i++)
          quantiles[i] = responses[random.Next(i0, i1)]; // sample randomly from all responses
      }
      else
      {
        // ...otherwise use all response values.
        nThresholds = i1 - i0 - 1;
        std::copy(&responses[i0], &responses[i1], quantiles.begin());
      }

      // Sort the response values to form approximate quantiles.
      std::sort(quantiles.begin(), quantiles.end());

      if (quantiles[0] == quantiles[nThresholds])
        return 0;   // all sampled response values were the same

      // Compute n candidate thresholds by sampling in between n+1 approximate quantiles
      for (int i = 0; i < nThresholds; i++)
        thresholds[i] = quantiles[i] + (float)(random_.NextDouble() * (quantiles[i + 1] - quantiles[i]));

      return nThresholds;
    }
  };

  
  /// <summary>
  /// Used for multi-threaded decision tree training. Candidate feature
  /// response function evaluation is distributed over multiple threads.
  /// </summary>
  template<class F, class S>
  class ParallelTreeTrainer
  {
  public:
    /// <summary>
    /// Train a new decision tree given some training data and a training
    /// problem described by an ITrainingContext instance.
    /// </summary>
    /// <param name="random">The single random number generator.</param>
    /// <param name="progress">Progress reporting target.</param>
    /// <param name="context">The ITrainingContext instance by which
    /// the training framework interacts with the training data.
    /// Implemented within client code.</param>
    /// <param name="parameters">Training parameters.</param>
    /// <param name="maxThreads">The maximum number of threads to use.</param>
    /// <param name="data">The training data.</param>
    /// <returns>A new decision tree.</returns>
    static std::auto_ptr<Tree<F, S> > TrainTree(
      Random& random,
      ITrainingContext<F, S>& context,
      const TrainingParameters& parameters,
      int maxThreads,
      const IDataPointCollection& data,
      ProgressStream* progress=0)
    {
      ProgressStream defaultProgress(std::cout, parameters.Verbose? Verbose:Interest);
      if(progress==0)
        progress=&defaultProgress;

      ParallelTreeTrainingOperation<F, S> trainingOperation(random, context, parameters, maxThreads, data, *progress);

      std::auto_ptr<Tree<F, S> > tree = std::auto_ptr<Tree<F, S> >(new Tree<F,S>(parameters.MaxDecisionLevels));

      (*progress)[Verbose] << std::endl;

      trainingOperation.TrainNodesRecurse(tree->GetNodes(), 0, 0, data.Count(), 0);  // will recurse until termination criterion is met

      (*progress)[Verbose] << std::endl;

      tree->CheckValid();

      return tree;
    }
  };

  /// <summary>
  /// Learns new decision forests from training data.
  /// </summary>
  template<class F, class S>
  class ParallelForestTrainer // where F:IFeatureResponse where S:IStatisticsAggregator<S>
  {
  public:
    /// <summary>
    /// Train a new decision forest given some training data and a training
    /// problem described by an instance of the ITrainingContext interface.
    /// </summary>
    /// <param name="random">Random number generator.</param>
    /// <param name="parameters">Training parameters.</param>
    /// <param name="context">An ITrainingContext instance describing
    /// the training problem, e.g. classification, density estimation, etc. </param>
    /// <param name="data">The training data.</param>
	/// <param name="progress">The progress.</param>
    /// <returns>A new decision forest.</returns>
    static std::auto_ptr<Forest<F,S> > TrainForest(
      Random& random,
      const TrainingParameters& parameters,
      ITrainingContext<F,S>& context,
      const IDataPointCollection& data,
      ProgressStream* progress=0)
    {
      ProgressStream defaultProgress(std::cout, parameters.Verbose? Verbose:Interest);
      if(progress==0)
        progress=&defaultProgress;

      std::auto_ptr<Forest<F,S> > forest = std::auto_ptr<Forest<F,S> >(new Forest<F,S>());

      for (int t = 0; t < parameters.NumberOfTrees; t++)
      {
        (*progress)[Interest] << "\rTraining tree "<< t << "...";

        std::auto_ptr<Tree<F, S> > tree = ParallelTreeTrainer<F, S>::TrainTree(random, context, parameters, data, progress);
        forest->AddTree(tree);
      }
      (*progress)[Interest] << "\rTrained " << parameters.NumberOfTrees << " trees.         " << std::endl;

      return forest;
    }
  };
} } }
