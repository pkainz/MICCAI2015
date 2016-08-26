/*
 * RandomForest.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef RANDOMFOREST_H_
#define RANDOMFOREST_H_

#include <vector>
#include <fstream>
#include "omp.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <eigen3/Eigen/Core>
#include "opencv2/opencv.hpp"

#include "RandomTree.h"
#include "DataSet.h"
#include "RFCoreParameters.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
class RandomForest
{
public:
    RandomForest(RFCoreParameters* hpin, AppContext* appcontextin);
    virtual ~RandomForest();

    virtual void Train(DataSet<Sample, Label>& dataset);
    virtual vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > Test(DataSet<Sample, Label>& dataset);
    virtual vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > Test(DataSet<Sample, Label>& dataset, int n_trees);
    virtual void Test(LabelledSample<Sample, Label>* sample, std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >& resulting_leafnodes);
    virtual void TestTree(LabelledSample<Sample, Label>* sample, std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >& resulting_leafnodes, int tree_id);
    virtual vector<LeafNodeStatistics> TestAndAverage(DataSet<Sample, Label>& dataset);
    virtual LeafNodeStatistics TestAndAverage(LabelledSample<Sample, Label>* sample);

    // stuff
    // TODO: this method should be removed and only the second one should be used! [DEPRICATED]
    void DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std); // only regression case
    void DenormalizeTargetVariables(std::vector<Eigen::VectorXd> mean, std::vector<Eigen::VectorXd> std); // joint classification-regression case

    // analysis tools
    vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > GetAllInternalNodes();
    vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > GetAllLeafNodes();

    virtual void Save(std::string savepath, int t_offset = 0);
    virtual void Load(std::string loadpath, int t_offset = 0);


    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // THIS METHOD IS NOT GENERIC !!! IT IS NEEDED IN THE SEMPRIORFOREST PROJECT
    // DO SOMETHING ELSE THERE ... DERIVE THE RF AND IMPLEMENT THIS METHOD THERE!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // clustering the leafnode statistics (only class histograms!) to get a new pseudo-class histogram
    // (but it only works with LeafnodeStatistics that have a m_class_histogram and m_pseudoclass_histogram
    // member variable!!!
    //virtual void CreatePseudoclassHistograms(bool copy_classes_to_pseudoclasses, int num_pseudoclasses, bool soft_assignment);


protected:

    // bagging methods
    void BaggingForTrees(DataSet<Sample, Label>& dataset_full, vector<DataSet<Sample, Label> >& dataset_inbag, vector<DataSet<Sample, Label> >& dataset_outbag);
    void SubsampleWithReplacement(DataSet<Sample, Label>& dataset_full, DataSet<Sample, Label>& dataset_inbag, DataSet<Sample, Label>& dataset_outbag);


    // parameters
	RFCoreParameters* m_hp;
	AppContext* m_appcontext;

	vector<RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>* > m_trees;

};



#include "RandomForest.cpp"

#endif /* RANDOMFOREST_H_ */

