/*
 * Node.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef NODE_H_
#define NODE_H_

#include <stdlib.h>

#include "DataSet.h"
#include "LabelledSample.h"
#include "RFCoreParameters.h"
#include <stdexcept>



namespace NODE_TYPE
{
    enum Enum
    {
    	SPLIT_NODE 				= 0,
    	INTERMEDIATE_LEAF 		= 1,
    	FINAL_LEAF 				= 2
    };
}

template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
class Node
{
public:
	// constructors
	Node(RFCoreParameters* hpin, AppContext* appcontextin);
    Node(RFCoreParameters* hpin, AppContext* appcontextin, int innodeid, int indepth, NODE_TYPE::Enum node_type, DataSet<Sample, Label>& dataset);
    virtual ~Node();

    int Split(LabelledSample<Sample, Label>* labelled_sample_in) const;

    void MakeSplitNode(SplitFunction *spfin);
    void MakeIntermediateLeaf();
    void MakeFinalLeaf();

    void CalcualteLeafNodeStatistics(DataSet<Sample, Label>& datasetin);
    void CalcualteLeafNodeStatistics();
    // This is for the refinement step (from bagging)
    void UpdateLeafnodeStatistics(LabelledSample<Sample, Label>* labelled_sample);

    // I/O methods
    virtual void Save(std::ofstream& out);
    virtual void Load(std::ifstream& in);


    // ####### members #####################################
    // node information
    int m_node_id;
	int m_depth;
    NODE_TYPE::Enum m_type;
    bool m_is_leaf;
	RFCoreParameters* m_hp;
	AppContext* m_appcontext;

    // data
    DataSet<Sample, Label> m_dataset;

    // For split nodes
    SplitFunction* m_splitfunction;

    // For leaf nodes
    LeafNodeStatistics* m_leafstats;
};

// a comparison function
template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
struct NodeCompare
{
    bool operator()(const Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* l, const Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* r)
    {
        return l->m_node_id < r->m_node_id;
    }
};



#include "Node.cpp"


#endif /* NODE_H_ */
