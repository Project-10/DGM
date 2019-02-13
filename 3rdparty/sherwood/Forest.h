// This file declares the Forest class, which is used to represent forests of decisions trees.
// Bug fixing by Sergey Kosov in 2016 for Project X
// C++17 support by Sergey Kosov in 2018 for Project X
#pragma once

#include <memory>
#include <stdexcept>
#include <fstream>
#include <istream>
#include <iostream>
#include <vector>

#include "ProgressStream.h"

#include "Interfaces.h"
#include "Tree.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
	/***
	* @brief A decision forest, i.e. a collection of decision trees.
	* @tparam F IFeatureResponse
	* @tparam S IStatisticsAggregator
	*/
	template<class F, class S> class Forest 
	{
	private:		
		static const char		* binaryFileHeader_;
		std::vector<Tree<F,S>*>	  trees_;


	public:
		typedef typename std::vector< Tree<F,S>* >::size_type TreeIndex;

		~Forest(void) 
		{
			for (TreeIndex t = 0; t < trees_.size(); t++) delete trees_[t];
		}
		/**
		* @brief Add another tree to the forest
		* @param tree The tree
		*/
		void AddTree(std::unique_ptr<Tree<F,S> > tree)
		{
			tree->CheckValid();
			trees_.push_back(tree.get());
			tree.release();
		}
		/**
		* @brief Deserialize a forest from a file
		* @param path The file path
		* @returns The forest
		*/
		static std::unique_ptr<Forest<F, S> > Deserialize(const std::string& path)
		{ 
			std::ifstream i(path.c_str(), std::ios_base::binary);
			return Forest<F,S>::Deserialize(i);
		}
		/**
		* @brief Deserialize a forest from a binary stream
		* @param i The stream
		* @returns
		*/
		static std::unique_ptr<Forest<F, S> > Deserialize(std::istream& i)
		{
			std::unique_ptr<Forest<F, S> > forest = std::unique_ptr<Forest<F, S> >(new Forest<F,S>());

			std::vector<char> buffer(strlen(binaryFileHeader_) + 1);
			i.read(&buffer[0], strlen(binaryFileHeader_));
			buffer[buffer.size() - 1] = '\0';

			if (strcmp(&buffer[0], binaryFileHeader_) != 0) {
				printf("%s !=\n%s\n", &buffer[0], binaryFileHeader_);
				throw std::runtime_error("Unsupported forest format.");
			}

			int majorVersion = 0, minorVersion = 0;
			i.read((char *) (&majorVersion), sizeof(majorVersion));
			i.read((char *) (&minorVersion), sizeof(minorVersion));

			if (majorVersion==0 && minorVersion==0) {
				size_t treeCount;
				i.read((char *) (&treeCount), sizeof(treeCount));

				for(size_t t = 0; t < treeCount; t++) {
					std::unique_ptr<Tree<F,S> > tree = Tree<F, S>::Deserialize(i);
					forest->trees_.push_back(tree.get());
					tree.release();
				}
			} else throw std::runtime_error("Unsupported file version number.");

			return forest;
		}
		/**
		* @brief Serialize the forest to file
		* @param path The file path
		*/
		void Serialize(const std::string &path)
		{
			std::ofstream o(path.c_str(), std::ios_base::binary);
			Serialize(o);
		}
		/**
		* @brief Serialize the forest a binary stream.
		* @param stream The stream
		*/
		void Serialize(std::ostream &stream)
		{
			const int majorVersion = 0, minorVersion = 0;
			stream.write(binaryFileHeader_, strlen(binaryFileHeader_));
			stream.write((const char *) (&majorVersion), sizeof(majorVersion));
			stream.write((const char *) (&minorVersion), sizeof(minorVersion));

			// NB. We could allow IFeatureResponse and IStatisticsAggregrator to write type information here for safer deserialization 
			// (and friendlier exception descriptions in the event that the user tries to deserialize a tree of the wrong type).

			size_t treeCount = TreeCount();
			stream.write((const char *) (&treeCount), sizeof(treeCount));

			for(size_t t = 0; t < TreeCount(); t++) GetTree((t)).Serialize(stream);

			if(stream.bad()) throw std::runtime_error("Forest serialization failed.");
		}
		/**
		* @brief Access the specified tree
		* @param index A zero-based integer index
		* @returns The tree
		*/
		const Tree<F,S>& GetTree(int index) const { return *trees_[index]; }
		/**
		* @brief Access the specified tree.
		* @param index A zero-based integer index.
		* @returns The tree.
		*/
		Tree<F,S> & GetTree(size_t index) { return *trees_[index]; }
		/**
		* @brief How many trees in the forest?
		*/
		size_t TreeCount(void) const { return trees_.size(); }
		/**
		* @brief Apply a forest of trees to a set of data points
		* @param data The data points
		* @param leafNodeIndices The indeces of leaf node
		* @param progress The progress
		*/
		void Apply(const IDataPointCollection &data, std::vector<std::vector<int> > &leafNodeIndices, ProgressStream *progress = 0) const
		{
			ProgressStream defaultProgressStream(std::cout, Interest);
			progress = (progress==0)?&defaultProgressStream:progress;

			leafNodeIndices.resize(TreeCount());

			for (size_t t = 0; t < TreeCount(); t++) {
				leafNodeIndices[t].resize(data.Count());

				//(*progress)[Interest] << "\rApplying tree " << t << "...";
				trees_[t]->Apply(data, leafNodeIndices[t]);
			}

			//(*progress)[Interest] << "\rApplied " << TreeCount() << " trees.        " << std::endl;
		}
	};

	template<class F, class S> const char * Forest<F,S>::binaryFileHeader_ = "MicrosoftResearch.Cambridge.Sherwood.Forest";
} } }
