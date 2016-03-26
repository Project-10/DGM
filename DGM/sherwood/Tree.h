#pragma once

// This file defines the Tree class, which is used to represent decision trees.

#include <assert.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "Interfaces.h"
#include "Node.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  template<class F, class S> class TreeTrainer;
  template<class F, class S> class ParallelTreeTrainer;

  /// <summary>
  /// A decision tree, comprising multiple nodes.
  /// </summary>
  template<class F, class S> 
  class Tree // where F:IFeatureResponse where S:IStatisticsAggregator<S>
  {
    static const char* binaryFileHeader_;

    typedef typename std::vector<unsigned int>::size_type DataPointIndex;

    int decisionLevels_;

    std::vector<Node<F,S> > nodes_;

  public:
    // Implementation only
    Tree(int decisionLevels):decisionLevels_(decisionLevels)
    {
      if(decisionLevels<0)
        throw std::runtime_error("Tree can't have less than 0 decision levels.");

      if(decisionLevels>19)
        throw std::runtime_error("Tree can't have more than 19 decision levels.");

      // This full allocation of node storage may be wasteful of memory
      // if trees are unbalanced but is efficient otherwise. Because child
      // node indices can determined directly from the parent node's index
      // it isn't necessary to store parent-child references within the
      // nodes.
      nodes_.resize((1 << (decisionLevels + 1)) - 1);
    }

    std::vector<Node<F,S> > & GetNodes() { return nodes_;}

  public:
    /// <summary>
    /// Apply the decision tree to a collection of test data points.
    /// </summary>
    /// <param name="data">The test data.</param>
	/// <param name="leafNodeIndices">The indices of leaf node.</param>
    /// <returns>An array of leaf node indices per data point.</returns>
    void Apply(const IDataPointCollection& data, std::vector<int>& leafNodeIndices)
    {
      CheckValid();

      leafNodeIndices.resize(data.Count()); // of leaf node reached per data point

      // Allocate temporary storage for data point indices and response values
      std::vector<size_t> dataIndices_(data.Count());
      for (unsigned int i = 0; i < data.Count(); i++)
        dataIndices_[i] = i;

      std::vector<float> responses_(data.Count());

      ApplyNode(0, data, dataIndices_, 0, (int) data.Count(), leafNodeIndices, responses_);
    }

	void Serialize(std::ostream& o) const
	{
		const int majorVersion = 0, minorVersion = 0;

		o.write(binaryFileHeader_, strlen(binaryFileHeader_));
		o.write((const char*)(&majorVersion), sizeof(majorVersion));
		o.write((const char*)(&minorVersion), sizeof(minorVersion));

		// NB. We could allow IFeatureResponse and IStatisticsAggregrator to
		// write type information here for safer deserialization (and
		// friendlier exception descriptions in the event that the user
		// tries to deserialize a tree of the wrong type).

		o.write((const char*)(&decisionLevels_), sizeof(decisionLevels_));

		for(size_t n=0; n<NodeCount(); n++)
			nodes_[n].Serialize(o);
	}

    static std::auto_ptr<Tree<F,S> > Deserialize(std::istream& i)
    {
      std::auto_ptr<Tree<F,S> > tree;

      std::vector<char> buffer(strlen(binaryFileHeader_)+1);
      i.read(&buffer[0], strlen(binaryFileHeader_));
      buffer[buffer.size()-1] = '\0';

	  if (strcmp(&buffer[0], binaryFileHeader_) != 0) {
		  printf("%s !=\n%s\n", &buffer[0], binaryFileHeader_);
		  throw std::runtime_error("Unsupported forest format.");
	  }

      int majorVersion = 0, minorVersion = 0;
      i.read((char*)(&majorVersion), sizeof(majorVersion));
      i.read((char*)(&minorVersion), sizeof(minorVersion));

      if(majorVersion==0 && minorVersion==0)
      {
        int decisionLevels;
        i.read((char*)(&decisionLevels), sizeof(decisionLevels));

        if(decisionLevels<=0)
          throw std::runtime_error("Invalid data");

        tree = std::auto_ptr<Tree<F,S> >(new Tree<F, S>(decisionLevels));

        for(size_t n = 0; n < tree->NodeCount(); n++)
          tree->nodes_[n].Deserialize(i);

        tree->CheckValid();
      }
      else
        throw std::runtime_error("Unsupported file version number.");

      return tree;
    }

    /// @brief The number of nodes in the tree, including decision, leaf, and null nodes.
    size_t NodeCount() const {return nodes_.size();}

    /// <summary>
    /// Return the specified tree node.
    /// </summary>
    /// <param name="index">A zero-based node index.</param>
    /// <returns>The node.</returns>
    const Node<F,S>& GetNode(int index) const
    {
      return nodes_[index];
    }

    /// <summary>
    /// Return the specified tree node.
    /// </summary>
    /// <param name="index">A zero-based node index.</param>
    /// <returns>The node.</returns>
    Node<F,S>& GetNode(int index)
    {
      return nodes_[index];
    }

    static DataPointIndex Partition(std::vector<float> &keys, std::vector<size_t> &values, DataPointIndex i0, DataPointIndex i1, float threshold)
    {
      assert(i1 > i0); // past-the-end element index must be greater than start element index.

      int i = (int)(i0);     // index of first element
      int j = int(i1 - 1); // index of last element

      while (i != j)
      {
        if (keys[i] >= threshold)
        {
          // Swap keys[i] with keys[j]
          float key = keys[i];
          size_t value = values[i];

          keys[i] = keys[j];
          values[i] = values[j];

          keys[j] = key;
          values[j] = value;

          j--;
        }
        else
        {
          i++;
        }
      }

      return keys[i] >= threshold ? i : i + 1;
    }

    void CheckValid() const
    {
      if(NodeCount()==0)
        throw std::runtime_error("Valid tree must have at least one node.");

      if(GetNode(0).IsNull()==true)
        throw std::runtime_error("A valid tree must have non-null root node.");

      CheckValidRecurse(0);
    }

  private:
    void CheckValidRecurse(int index, bool bHaveReachedLeaf=false) const
    {
      if (bHaveReachedLeaf==false && GetNode(index).IsLeaf())
      {
        // First time I have encountered a leaf node
        bHaveReachedLeaf = true;
      }
      else
      {
        if (bHaveReachedLeaf)
        {
          // Have encountered a leaf node already, this node had better be null
          if (GetNode(index).IsNull() == false)
            throw std::runtime_error("Valid tree must have all descendents of leaf nodes set as null nodes.");
        }
        else
        {
          // Have not encountered a leaf node yet, this node had better be a split node
          if (GetNode(index).IsSplit() == false)
            throw std::runtime_error("Valid tree must have all antecents of leaf nodes set as split nodes.");
        }
      }

      if (index >= ((int) NodeCount() - 1) / 2)
      {
        // At maximum depth, this node had better be a leaf
        if (bHaveReachedLeaf == false)
          throw std::runtime_error("Valid tree must have all branches terminated by leaf nodes.");
      }
      else
      {
        CheckValidRecurse(2 * index + 1, bHaveReachedLeaf);
        CheckValidRecurse(2 * index + 2, bHaveReachedLeaf);
      }
    }

  public:
    static std::string GetPrettyPrintPrefix(int nodeIndex)
    {
      std::string prefix = nodeIndex > 0 ? (nodeIndex % 2 == 1 ? "|-o " : "+-o ") : "o ";
      for (int l = (nodeIndex - 1) / 2; l > 0; l = (l - 1) / 2)
        prefix = (l % 2 == 1 ? "| " : "  ") + prefix;
      return prefix;
    }

  private:
    void ApplyNode(int nodeIndex, const IDataPointCollection &data, std::vector<size_t> &dataIndices, int i0, int i1, std::vector<int> &leafNodeIndices, std::vector<float>& responses_)
    {
      assert(nodes_[nodeIndex].IsNull()==false);

      Node<F,S>& node = nodes_[nodeIndex];

      if (node.IsLeaf())
      {
        for (int i = i0; i < i1; i++)
          leafNodeIndices[dataIndices[i]] = nodeIndex;
        return;
      }

      if (i0 == i1)   // No samples left
        return;

      for (int i = i0; i < i1; i++) 
		responses_[i] = node.Feature.GetResponse(data, dataIndices[i]);
	  

      int ii = (int) Partition(responses_, dataIndices, i0, i1, node.Threshold);

      // Recurse for child nodes.
      ApplyNode(nodeIndex * 2 + 1, data, dataIndices, i0, ii, leafNodeIndices, responses_);
      ApplyNode(nodeIndex * 2 + 2, data, dataIndices, ii, i1, leafNodeIndices, responses_);
    }
  };

  template<class F, class S>
  const char* Tree<F,S>::binaryFileHeader_ = "MicrosoftResearch.Cambridge.Sherwood.Tree";
} } }

