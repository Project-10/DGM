#include "IGraphPairwise.h"

namespace DirectGraphicalModels
{
    void IGraphPairwise::addEdge(size_t srcNode, size_t dstNode, const Mat &pot)
    {
        addEdge(srcNode, dstNode, 0, pot);
    }
    
    bool IGraphPairwise::isEdgeArc(size_t srcNode, size_t dstNode) const
    {
        return isEdgeExists(dstNode, srcNode);
    }
   
    // Add a new (undirected edge) ark to the graph with specified potentional
    void IGraphPairwise::addArc(size_t Node1, size_t Node2, const Mat &pot)
    {
        addArc(Node1, Node2, 0, pot);
    }
    
    // Add a new (undirected edge) ark to the graph with specified potentional
	void IGraphPairwise::addArc(size_t Node1, size_t Node2, byte group, const Mat &pot)
	{
		if (pot.empty()) {
			addEdge(Node1, Node2, group, Mat());
			addEdge(Node2, Node1, group, Mat());
		}
		else {
			Mat Pot;
			sqrt(pot, Pot);
			addEdge(Node1, Node2, group, Pot);
			addEdge(Node2, Node1, group, Pot.t());
		}
	}

	// Add a new (undirected edge) arc to the graph with specified potentional
	void IGraphPairwise::setArc(size_t Node1, size_t Node2, const Mat &pot)
	{
		Mat Pot;
		sqrt(pot, Pot);
		setEdge(Node1, Node2, Pot);
		setEdge(Node2, Node1, Pot.t());
	}
    
    void IGraphPairwise::setArcGroup(size_t Node1, size_t Node2, byte group)
    {
        setEdgeGroup(Node1, Node2, group);
        setEdgeGroup(Node2, Node1, group);
    }
    
    void IGraphPairwise::removeArc(size_t Node1, size_t Node2)
    {
        removeEdge(Node1, Node2);
        removeEdge(Node2, Node1);
    }
    
    bool IGraphPairwise::isArcExists(size_t Node1, size_t Node2) const
    {
        return (isEdgeExists(Node1, Node2) && isEdgeExists(Node2, Node1));
    }
	
	void IGraphPairwise::marginalize(const vec_size_t &nodes)
    {
        Mat pot, pot1, pot2;
        
        for (size_t node : nodes) {
            vec_size_t parentNodes, childNodes, managers;
            getParentNodes(node, parentNodes);
            getChildNodes(node, childNodes);
            
            // find all managers for the node
            for (size_t child : childNodes) {
                // Looking for those child nodes, which are managers
                auto isArc = std::find(parentNodes.begin(), parentNodes.end(), child);    // If there is a return edge => the child is a neighbor
                if (isArc != parentNodes.end()) continue;
                // Here the child is a manager
                auto isInZ = std::find(nodes.begin(), nodes.end(), child);                // If the manager is to be also marginalized
                if (isInZ != nodes.end()) continue;
                
                managers.push_back(child);
                
                // Add new edges (from any other neighboring node to the manager)
                for (size_t parent : parentNodes) {
                    auto isInZ = std::find(nodes.begin(), nodes.end(), parent);            // If the parent is to be also marginalized
                    if (isInZ != nodes.end()) continue;
                    
                    getEdge(parent, node, pot1);
                    getEdge(node, child, pot2);
                    if (pot1.empty() && pot2.empty()) addEdge(parent, child);
                    else {
                        pot1.empty() ? pot = pot2 + pot1 : pot = pot1 + pot2;
                        addEdge(parent, child, pot);
                    }
                }
            }
            
            // Add new arcs (between two managers)
            if (managers.size() >= 2)
                for (size_t i = 0; i < managers.size() - 1; i++)
                    for (size_t j = i + 1; j < managers.size(); j++) {
                        getEdge(node, managers[i], pot1);
                        getEdge(node, managers[j], pot2);
                        if (pot1.empty() && pot2.empty()) addArc(managers[i], managers[j]);
                        else {
                            pot1.empty() ? pot = pot2 + pot1 : pot = pot1 + pot2;
                            addArc(managers[i], managers[j], pot);
                        }
                    }
            
            // Delete all
            for (size_t &parent : parentNodes) removeEdge(parent, node);
            for (size_t &child : childNodes)   removeEdge(node, child);
        } // n
    }
}
