/*
 * RandomForest.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef RANDOMFOREST_CPP_
#define RANDOMFOREST_CPP_

#include "RandomForest.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::RandomForest(RFCoreParameters* hpin, AppContext* appcontextin) : m_hp(hpin), m_appcontext(appcontextin)
{
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::~RandomForest()
{
    // Free the trees
    for (int i = 0; i < m_trees.size(); i++)
        delete(this->m_trees[i]);
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Train(DataSet<Sample, Label>& dataset)
{
	// This is the standard random forest training procedure ...
    vector<DataSet<Sample, Label> > inbag_dataset(m_hp->m_num_trees), outbag_dataset(m_hp->m_num_trees);
	this->BaggingForTrees(dataset, inbag_dataset, outbag_dataset);

	// Train the trees
	m_trees.resize(m_hp->m_num_trees);
	for (int t = 0; t < m_hp->m_num_trees; t++)
	{
		m_trees[t] = new RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(m_hp, m_appcontext);
		m_trees[t]->Init(inbag_dataset[t]);
	}

	for (unsigned int d = 0; d < m_hp->m_max_tree_depth; d++)
	{
		int num_nodes_left = 0;
		for (int t = 0; t < m_hp->m_num_trees; t++)
			num_nodes_left += m_trees[t]->GetTrainingQueueSize();
		if (!m_hp->m_quiet)
			std::cout << "RF: training depth " << d << " of the forest -> " << num_nodes_left << " nodes left for splitting" << std::endl;

		// train the trees
        #pragma omp parallel for
		for (int t = 0; t < m_hp->m_num_trees; t++)
			m_trees[t]->Train(d);
	}
	if (m_hp->m_do_tree_refinement)
	{
        #pragma omp parallel for
		for (int t = 0; t < m_hp->m_num_trees; t++)
			m_trees[t]->UpdateLeafStatistics(outbag_dataset[t]);
	}
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>*> >
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Test(DataSet<Sample, Label>& dataset)
{
	return this->Test(dataset, (int)this->m_trees.size());
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > >
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Test(DataSet<Sample, Label>& dataset, int num_trees_to_test)
{
	if (num_trees_to_test < 1 || num_trees_to_test > (int)m_trees.size())
		num_trees_to_test = (int)m_trees.size();

	vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > tmp(num_trees_to_test);
	vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > resulting_leafnodes((int)dataset.size(), tmp);
    #pragma omp parallel for
	for (int t = 0; t < num_trees_to_test; t++)
	{
		for (int s = 0; s < (int)dataset.size(); s++)
		{
			resulting_leafnodes[s][t] = m_trees[t]->Test(dataset[s]);
		}
	}
    return resulting_leafnodes;
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Test(LabelledSample<Sample, Label>* sample, std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >& resulting_leafnodes)
{
	resulting_leafnodes.resize(m_trees.size());
    for (size_t t = 0; t < m_trees.size(); t++)
    	resulting_leafnodes[t] = m_trees[t]->Test(sample);
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::TestTree(LabelledSample<Sample, Label>* sample, std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >& resulting_leafnodes, int tree_id)
{
	resulting_leafnodes.resize(1);
	resulting_leafnodes[0] = m_trees[tree_id]->Test(sample);
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
std::vector<LeafNodeStatistics>
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::TestAndAverage(DataSet<Sample, Label>& dataset)
{
	vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > leafnodes = this->Test(dataset);
	vector<LeafNodeStatistics> ret_stats(dataset.size(), this->m_appcontext);
    #pragma omp parallel for
	for (size_t s = 0; s < dataset.size(); s++)
	{
		vector<LeafNodeStatistics*> tmp_stats(leafnodes[s].size());
		for (size_t t = 0; t < leafnodes[s].size(); t++)
		{
			tmp_stats[t] = leafnodes[s][t]->m_leafstats;
		}
		ret_stats[s] = LeafNodeStatistics::Average(tmp_stats, this->m_appcontext);
	}
	return ret_stats;
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
LeafNodeStatistics
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::TestAndAverage(LabelledSample<Sample, Label>* sample)
{
	vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > leafnodes;
	this->Test(sample, leafnodes);

	std::vector<LeafNodeStatistics*> tmp_stats(leafnodes.size());
	for (size_t t = 0; t < leafnodes.size(); t++)
		tmp_stats[t] = leafnodes[t]->m_leafstats;
	return LeafNodeStatistics::Average(tmp_stats, this->m_appcontext);
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std)
{
	// get all leaf nodes
	vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > leafs;
	leafs = this->GetAllLeafNodes();

	// call DenormalizeTargetVariables method in each of the leafnodes-statistics
	for (size_t t = 0; t < leafs.size(); t++)
	{
		for (size_t i = 0; i < leafs[t].size(); i++)
		{
			leafs[t][i]->m_leafstats->DenormalizeTargetVariables(mean, std);
		}
	}
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::DenormalizeTargetVariables(std::vector<Eigen::VectorXd> mean, std::vector<Eigen::VectorXd> std)
{
	// get all leaf nodes
	vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > leafs;
	leafs = this->GetAllLeafNodes();

	// call DenormalizeTargetVariables method in each of the leafnodes-statistics
	for (size_t t = 0; t < leafs.size(); t++)
	{
		for (size_t i = 0; i < leafs[t].size(); i++)
		{
			leafs[t][i]->m_leafstats->DenormalizeTargetVariables(mean, std);
		}
	}
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > >
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GetAllInternalNodes()
{
	std::vector<std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > ret(this->m_trees.size());
	for (size_t t = 0; t < this->m_trees.size(); t++)
	{
		ret[t] = this->m_trees[t]->GetAllInternalNodes();
	}
	return ret;
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > >
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GetAllLeafNodes()
{
	std::vector<std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > ret(this->m_trees.size());
	for (size_t t = 0; t < this->m_trees.size(); t++)
	{
		ret[t] = this->m_trees[t]->GetAllLeafNodes();
	}
	return ret;
}




template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Save(std::string savepath, int t_offset)
{
	// This function should be rather called "SaveTrees", as no information on the complete forest is stored
	// this has the benefits that one can train 100 trees, but then use only a subset for testing, if desired
	// e.g., for analysis of the parameter #trees

	// check if storage folder exists. If not, try to create the folder.
	struct stat info;
	if (stat(savepath.c_str(), &info) != 0)
	{
	    int status = mkdir(savepath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (status == -1)
		{
			std::cout << "Could not create the folder to store trees" << std::endl;
			throw std::runtime_error("Could not create savepath");
		}
	}
	else if (info.st_mode & S_IFDIR)
	{
		// everything is ok
	}
	else
	{
		std::cout << savepath << " is not a folder" << std::endl;
		throw std::runtime_error("not a folder");
	}

	// iterate over all trees
	for (int t = 0; t < m_trees.size(); t++)
	{
		std::stringstream tree_savefile_stream;
		tree_savefile_stream << savepath << "tree_" << t + t_offset << ".txt";
		std::string tree_savefile = tree_savefile_stream.str();
		std::ofstream out(tree_savefile.c_str(), ios::binary);
		// TODO: we could try to write a real binary file with out.write(...,...). Maybe the files become smaller
		m_trees[t]->Save(out);
		out.flush();
		out.close();
	}
}

/**
 * Loading a random forest from a set of .txt files stored in a folder. The number of trees loaded depends
 * on the number of trees defined in the config file. Each tree has to be stored in a separate .txt file
 * named "tree_X.txt", where X goes from to 0 to #trees-1. One can also use an offset value to start from
 * any tree > 0.
 *
 * @params[in] loadpath path to the folder the trees are stored
 * @params[in] t_offset the offset of the trees to start loading
 */
template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Load(std::string loadpath, int t_offset)
{
	m_trees.resize(m_hp->m_num_trees);
	for (int t = 0; t < m_hp->m_num_trees; t++)
	{
		// create a new tree
		m_trees[t] = new RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(m_hp, m_appcontext);

		// load the tree
		std::stringstream tree_loadfile_stream;
		tree_loadfile_stream << loadpath << "tree_" << t + t_offset << ".txt";
		std::string tree_loadfile = tree_loadfile_stream.str();
		std::ifstream in(tree_loadfile.c_str(), ios::in);
		m_trees[t]->Load(in);
		in.close();
	}
}




// ############### PRIVATE METHODS #######################

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::BaggingForTrees(DataSet<Sample, Label>& dataset_full, vector<DataSet<Sample, Label> >& dataset_inbag, vector<DataSet<Sample, Label> >& dataset_outbag)
{
	if ((int)dataset_inbag.size() != m_hp->m_num_trees)
		dataset_inbag.resize(m_hp->m_num_trees);
	if ((int)dataset_outbag.size() != m_hp->m_num_trees)
		dataset_outbag.resize(m_hp->m_num_trees);

	size_t fixed_n;
	vector<int> rand_inds;
	for (unsigned int t = 0; t < m_hp->m_num_trees; t++)
    {
		DataSet<Sample, Label> temp1, temp2;
        switch (m_hp->m_bagging_method)
        {
        case TREE_BAGGING_TYPE::NONE:
			dataset_inbag[t] = dataset_full;
			// no out-of-bag samples
			break;
        case TREE_BAGGING_TYPE::SUBSAMPLE_WITH_REPLACEMENT:
            this->SubsampleWithReplacement(dataset_full, temp1, temp2);
            dataset_inbag[t] = temp1;
            dataset_outbag[t] = temp2;
            break;
        case TREE_BAGGING_TYPE::FIXED_RANDOM_SUBSET:
        	fixed_n = min((size_t)3000, dataset_full.size());
        	rand_inds = randPermSTL((int)dataset_full.size());
        	dataset_inbag[t].Resize(fixed_n);
        	for (size_t i = 0; i < fixed_n; i++)
        		dataset_inbag[t].SetLabelledSample(i, dataset_full[rand_inds[i]]);
        	// no out-of-bag samples
        	break;
        default:
        	throw std::runtime_error("RandomForest: wrong sampling method defined!");
            break;
        }
    }
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::SubsampleWithReplacement(DataSet<Sample, Label>& dataset_full, DataSet<Sample, Label>& dataset_inbag, DataSet<Sample, Label>& dataset_outbag)
{
	int num_total_samples = (int)dataset_full.size();
	vector<int> inIndex(num_total_samples, 0);
	vector<int> usedSamples(num_total_samples, 0);
    for (int n = 0; n < num_total_samples; n++)
    {
        // get a random index
        inIndex[n] = (int)floor(num_total_samples * randDouble());
        if (usedSamples[inIndex[n]] == 0)
        {
        	dataset_inbag.AddLabelledSample(dataset_full[(unsigned int)inIndex[n]]);
        	usedSamples[inIndex[n]] = 1;
        }
    }
    // set the outbag samples
    for (int n = 0; n < num_total_samples; n++)
    {
        if (usedSamples[n] == 0)
        {
        	dataset_outbag.AddLabelledSample(dataset_full[(unsigned int)n]);
        }
    }
}




#endif /* RANDOMFOREST_CPP_ */
