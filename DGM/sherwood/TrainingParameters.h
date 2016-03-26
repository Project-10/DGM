#pragma once

// This file defines the ParallelParallelTreeTraininer class, which is responsible for
// creating new Tree instances by learning from training data.

#include <assert.h>

#include <vector>
#include <string>
#include <algorithm>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  /// <summary>
  /// Decision tree training parameters.
  /// </summary>
  struct TrainingParameters
  {
    TrainingParameters()
    {
      // Some sane defaults will need to be changed per application.
      NumberOfTrees = 1;
      NumberOfCandidateFeatures = 10;
      NumberOfCandidateThresholdsPerFeature = 10;
      MaxDecisionLevels = 5;
      Verbose = false;
    }

    int NumberOfTrees;
    int NumberOfCandidateFeatures;
    unsigned int NumberOfCandidateThresholdsPerFeature;
    int MaxDecisionLevels;
    bool Verbose;
  };
} } }