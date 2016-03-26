#pragma once

// This file defines interfaces used during decision forest training and
// evaluation. These interfaces are intended to be implemented within client
// code.
//
// *** NOTE that the template-based C++ version of the framework does not
// require that IFeatureResponse and IStatisticsAggregator implementations
// are actually derived from the corresponding abstract base classes. This is
// because concrete type information is supplied at compile time in the form
// of template type arguments. These abstract base classes merely describe the
// (compile time) contract that concrete IFeatureResponse and
// IStatisticsAggregator implementations must fulfil. Implementors will
// typically choose NOT to derive from these abstract base classes to avoid
// the memory and performance overhead of a virtual function table pointer.

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  /// <summary>
  /// A collection of data points used for forest training or evaluation.
  /// Concrete implementations supplied by client code will collaborate
  /// with concrete IFeatureResponse and IStatisticsAggregator implementations for
  /// feature evaluation and statistics aggregation over data points.
  /// </summary>
  class IDataPointCollection
  {
  public:
    virtual ~IDataPointCollection(void) {};
    virtual size_t Count(void) const=0;
  };

  /// <summary>
  /// Features compute (single precision) response values for data points. A
  /// 'weak learner' comprises a feature and an associated decision threshold.
  /// </summary>
  class IFeatureResponse
  {
  public:
    /// <summary>
    /// Computes the response for the specified data point.
    /// </summary>
    /// <param name="data">The data.</param>
    /// <param name="dataIndex">The index of the data point to be evaluated.</param>
    /// <returns>A single precision response value.</returns>
    virtual float GetResponse(const IDataPointCollection& data, size_t dataIndex) const=0;
  };

  /// <summary>
  /// Used during forest training to aggregate statistics over sets of data
  /// points. The precise nature of the statistic to be aggregated is up to
  /// the caller. Common statistics include histograms over class labels (for
  /// classification problems) and sum and sum of squares (for regression
  /// problems).
  /// </summary>
  template<class S>
  class IStatisticsAggregator // where S IStatisticsAggregator
  {
  public:
    /// <summary>
    /// Called by the training framework to reset sample statistics. Allows
    /// IStatisticsAggregrator instances to be reused in the interests of
    /// avoiding unnecessary memeory allocations.
    /// </summary>
    virtual void Clear()=0;

    /// <summary>
    /// Update statistics with one additional data point.
    /// </summary>
    /// <param name="data">The data point collection.</param>
    /// <param name="index">The index of the data point.</param>
    virtual void Aggregate(const IDataPointCollection& data, size_t index)=0;

    /// <summary>
    /// Combine two sets of statistics.
    /// </summary>
    /// <param name="i">The statistics to be combined.</param>
    virtual void Aggregate(const S& i)=0;

    /// <summary>
    /// Called by the training framework to make a clone of the sample statistics to be stored in the leaf of a tree
    /// </summary>
    /// <returns></returns>
    virtual S DeepClone() const=0;
  };

  /// <summary>
  /// An abstract representation of a decision forest training problem that
  /// intended to be implemented within client code. Instances of this
  /// interface are used by the training framework to instantiate new
  /// IFeatureResponse and IStatisticsAggregator instances, to compute
  /// information gain, and to decide when to terminate training of a
  /// particular tree branch.
  /// </summary>
  template<class F, class S>
  class ITrainingContext //  where F:IFeatureResponse where S:IStatisticsAggregator<S>
  {
  public:
    /// <summary>
    /// Called by the training framework to generate a new random feature.
    /// Concrete implementations must return a new feature.
    /// </summary>
    virtual F GetRandomFeature(Random& random) = 0;

    /// <summary> 
    /// Called by the training framework to get an instance of
    /// a concrete IStatisticsAggregator implementation.
    /// </summary>
    virtual S GetStatisticsAggregator() = 0;

    /// <summary>
    /// Called by the training framework to compute the gain over a given
    /// binary partition of a set of samples.
    /// </summary>
    /// <param name="parent">Statistics aggregated over the complete set of samples.</param>
    /// <param name="leftChild">Statistics aggregated over the left hand partition.</param>
    /// <param name="rightChild">Statistics aggregated over the right hand partition.</param>
    /// <returns>A measure of gain, e.g. entropy gain in bits.</returns>
    virtual double ComputeInformationGain(const S& parent, const S& leftChild, const S& rightChild) = 0;

    /// <summary>
    /// Called by the training framework to determine whether training
    /// should terminate for this branch.  Concrete implementations must
    /// determine whether to terminate training based on statistics
    /// aggregated over the left and right hand sides of the best
    /// binary partition found.
    /// </summary>
    /// <param name="parent">Statistics aggregated over the complete set of samples.</param>
    /// <param name="leftChild">Statistics aggregated over the left hand partition.</param>
    /// <param name="rightChild">Statistics aggregated over the right hand partition.</param>
    /// <param name="gain">Gain computed for this binary partition
    /// within a previous call to ISampleCollection.ComputeGain().</param>
    /// <returns>True if training should be terminated, false otherwise.</returns>
    virtual bool ShouldTerminate(const S& parent, const S& leftChild, const S& rightChild, double gain) = 0;
  };
} } }
