/*
 * RandomTree.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef RANDOMTREE_H_
#define RANDOMTREE_H_

#include <vector>
#include <algorithm>
#include <fstream>

#include "Utilities.h"
#include "LabelledSample.h"
#include "Node.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
class RandomTree
{
public:

	// Constructors & Destructors
    RandomTree(RFCoreParameters* hpin, AppContext* appcontextin);
    virtual ~RandomTree();

    // Training & Testing methods
    virtual void Init(DataSet<Sample, Label>& dataset);
    virtual void Train(int depth);
    virtual void Train(LabelledSample<Sample, Label>* sample); // for online versions...
    virtual Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* Test(LabelledSample<Sample, Label>* labelled_sample);

    // Helper & Status methods
    int GetTrainingQueueSize() { return (int)this->m_pq.size(); }
    void UpdateLeafStatistics(DataSet<Sample, Label>& dataset);

    // Analysis tools
    std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > GetAllInternalNodes();
    std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > GetAllLeafNodes();
    std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > GetInternalNodesForSample(LabelledSample<Sample, Label>* labelled_sample);

    // I/O methods
    virtual void Save(std::ofstream& out);
    virtual void Load(std::ifstream& in);

    bool m_is_ADFTree;
    int m_prediction_type_ADF;


protected:

    // general tree building functions
    void GrowPriority(int num_splits);
    void SplitNode(SplitFunction* spf, DataSet<Sample, Label>& dataset_parent, DataSet<Sample, Label>& dataset_leftChild, DataSet<Sample, Label>& dataset_rightChild);
    virtual void GetRandomSampleSubsetForSplitting(DataSet<Sample, Label>& dataset_full, DataSet<Sample, Label>& dataset_subsample, int samples_per_class);

    // find splitting functions (classification + regression)
    virtual bool FindSplittingFunction(SplitFunction* spf, SplitEvaluator* spe, DataSet<Sample, Label>& dataset_parent, int depth, int node_id);


    // variables for storing the tree structure and tree data, etc.
    MatrixXd m_treetable;

    // width of the treetable
    int m_split_store_size;

    vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > m_nodes;
    vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > m_pq;
    RFCoreParameters* m_hp;
    AppContext* m_appcontext;

    // number of nodes in each depth
    vector<int> m_nodes_in_depth;

    // (current) number of leaf nodes of this tree
    int m_num_leafs;
    
    // max number of nodes (pow(2, max_tree_depth+1)-1)
    int m_num_nodes;
};



#include "RandomTree.cpp"

#endif /* RANDOMTREE_H_ */
