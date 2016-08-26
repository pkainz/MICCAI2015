/*
 * RandomTree.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef RANDOMTREE_CPP_
#define RANDOMTREE_CPP_

#include "RandomTree.h"




template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::RandomTree(RFCoreParameters* hpin, AppContext* appcontextin) :
		m_hp(hpin), m_appcontext(appcontextin), m_num_nodes(0), m_num_leafs(0), m_split_store_size(3),
		m_is_ADFTree(false), m_prediction_type_ADF(-1)
{
	// Treetable illustration:
	// width of table = 3
	// 0) depth
	// IF 		node is a splitnode:
	// 1-2) left and right child node id (within this treetable matrix)
	// ELSE		node is a leaf node:
	// 1-2) [0 0]
	// the id of the nodes is the row in this table, and is also stored in the node object itself!
}



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::~RandomTree()
{
	// free memory, delete all nodes
	for (int i = 0; i < this->m_nodes.size(); i++)
		delete(m_nodes[i]);
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Init(DataSet<Sample, Label>& dataset)
{
    // add the root node to the current tree
	m_treetable = MatrixXd::Zero(1, m_split_store_size); // depth, childLeftID, childRightID
    m_num_leafs = 1;
    m_num_nodes = 1;

    Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* rootnode = new Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>(m_hp, m_appcontext, 0, 0, NODE_TYPE::INTERMEDIATE_LEAF, dataset);
    m_nodes.push_back(rootnode);

    // put this node in the priority queue (it is considered for splitting)
    m_pq.push_back(rootnode);

    // init the <nodes_in_depth> structure
    m_nodes_in_depth.resize(m_hp->m_max_tree_depth, 0); // max_tree_depth x 1 vector with all zeros at start
    // set the number of nodes to be split in the first iteration (i.e., depth 0 has only the rootnode)
    m_nodes_in_depth[0] = 1;

    if (m_hp->m_debug_on)
    {
        cout << "Tree::Init()" << endl;
        cout << m_treetable << endl;
        cout << m_num_leafs << ", " << m_num_nodes << ", " << dataset.size() << ", " << m_nodes.size() << endl;
    }
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Train(int depth)
{
    // TODO: proper implementation of this chunk-like training ... switch between the methods!

    // Grow the tree
    int num_nodes_in_current_depth = m_nodes_in_depth[depth];
    this->GrowPriority(num_nodes_in_current_depth);
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Train(LabelledSample<Sample, Label>* sample)
{
    throw std::logic_error("Caution: training with a single labelled sample is not implemented in the base class!");
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>*
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Test(LabelledSample<Sample, Label>* labelled_sample)
{
	int node_id = 0;
	while(!this->m_nodes[node_id]->m_is_leaf)
	{
		if (this->m_nodes[node_id]->Split(labelled_sample) == 0)
			node_id = m_treetable(node_id, 1);
		else
			node_id = m_treetable(node_id, 2);
	}
	return this->m_nodes[node_id];
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::UpdateLeafStatistics(DataSet<Sample, Label>& dataset)
{
	for (int s = 0; s < dataset.size(); s++)
    {
    	// get the current labelled sample
    	LabelledSample<Sample, Label>* labelled_sample = dataset[s];

    	// route it to the corresponding leaf node
    	int node_id = 0;
        while(!this->m_nodes[node_id]->m_is_leaf)
		{
			if (this->m_nodes[node_id]->Split(labelled_sample) == 0)
				node_id = (int)m_treetable(node_id, 1);
			else
				node_id = (int)m_treetable(node_id, 2);
		}

		// update the leaf node statistics
		this->m_nodes[node_id]->UpdateLeafnodeStatistics(labelled_sample);
    }
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GetAllInternalNodes()
{
	std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > ret;
	for (size_t i = 0; i < this->m_nodes.size(); i++)
	{
		if (!this->m_nodes[i]->m_is_leaf)
			ret.push_back(this->m_nodes[i]);
	}
	return ret;
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GetInternalNodesForSample(LabelledSample<Sample, Label>* labelled_sample)
{
	std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > ret;
	// route it to the corresponding leaf node
	int node_id = 0;
	while(!this->m_nodes[node_id]->m_is_leaf)
	{
		// add the split node
		ret.push_back(this->m_nodes[node_id]);

		if (this->m_nodes[node_id]->Split(labelled_sample) == 0)
			node_id = (int)m_treetable(node_id, 1);
		else
			node_id = (int)m_treetable(node_id, 2);
	}

	return ret;
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* >
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GetAllLeafNodes()
{
	std::vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > ret;
	for (size_t i = 0; i < this->m_nodes.size(); i++)
	{
		if (this->m_nodes[i]->m_is_leaf)
			ret.push_back(this->m_nodes[i]);
	}
	return ret;
}


// I/O methods
template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Save(std::ofstream& out)
{
	// 1) store the treetable
	out << m_num_nodes << " " << m_num_leafs << endl;
	out << m_treetable.rows() << " " << m_treetable.cols() << endl;
	out << m_treetable << endl;

	// 2) store the nodes
	out << m_nodes.size() << endl;
	for (int n = 0; n < m_nodes.size(); n++)
		m_nodes[n]->Save(out);
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Load(std::ifstream& in)
{
	// 1) load the treetable
	in >> this->m_num_nodes >> this->m_num_leafs;
	int treetable_rows;
	in >> treetable_rows >> this->m_split_store_size;
	m_treetable = Eigen::MatrixXd::Zero(treetable_rows, this->m_split_store_size);
	for (int r = 0; r < m_treetable.rows(); r++)
	{
		for (int c = 0; c < m_treetable.cols(); c++)
			in >> m_treetable(r, c);
	}

	// 2) load the nodes
	in >> m_num_nodes;
	m_nodes.resize(m_num_nodes);
	for (int n = 0; n < m_nodes.size(); n++)
	{
		m_nodes[n] = new Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>(m_hp, m_appcontext);
		m_nodes[n]->Load(in);
	}

}






// PROTECTED methods
template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GrowPriority(int num_splits)
{
    // This method now works with "chunks"
    // It uses a priority queue to select the next nodes to be split
    //   ... the order can be defined (breadth-first, depth-first, etc ...)
    // Also, this function only splits <num_splits> nodes:
    //
    // Eg: - Use breadth-first and num_nodes_in_depth d to grow only nodes in depth d of the tree!
    //     - Use depth/breadth-first and use max_num_nodes in tree (or any higher number) to train a standard depth/breadth first tree
    //     - Use depth/breadth-first and use num_splits = 1 to only grow node-by-node ...
    //
    // After each chunk, you will return to the HoughForest class -> train function -> there you can control how to grow the tree!!

    if (m_hp->m_debug_on)
        cout << "Queue size: " << m_pq.size() <<  " | num_splits = " << num_splits << endl;

    int cnt = 0;
    while (!this->m_pq.empty() && cnt < num_splits)
    {
        // increase the split nodes counter
        cnt++;

        // Get the current leaf node to be considered for splitting
        Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* cnode = m_pq[0];
        m_pq.erase(m_pq.begin()); // only make the vector smaller
        int cnode_id = cnode->m_node_id;
        int cnode_depth = cnode->m_depth;
        // Get all samples for the current node
        DataSet<Sample, Label> cnode_dataset = cnode->m_dataset;

        // Some Debug outputs
        if (m_hp->m_debug_on)
        {
            cout << "Node " << cnode_id << " in depth " << cnode_depth << " is considered for splitting, having " << (int)cnode_dataset.size() << " samples to split" << endl;
        }

        // Check if we want to further split the node or not
        SplitEvaluator* spe = new SplitEvaluator(m_appcontext, cnode_depth, cnode_dataset);
        if (spe->DoFurtherSplitting(cnode_dataset, cnode_depth))
        {
        	if (m_hp->m_debug_on)
                cout << "Try to split node " << cnode_id << " at depth " << cnode_depth << endl << flush;

        	// create a new potential split function
        	SplitFunction* spf = new SplitFunction(m_appcontext);

            // Get samples for optimization (e.g., via random subsampling)
            DataSet<Sample, Label> dataset_for_optimization = cnode_dataset;
            if (m_hp->m_num_random_samples_for_splitting > 0)
                GetRandomSampleSubsetForSplitting(cnode_dataset, dataset_for_optimization, m_hp->m_num_random_samples_for_splitting);

            if (m_hp->m_debug_on)
                cout << "Try to find a splitting function ... " << flush;

            // find a good splitting function for this node
            bool foundSplit = this->FindSplittingFunction(spf, spe, dataset_for_optimization, cnode_depth, cnode_id);

            // If no split function was found, make a leaf
            if (!foundSplit)
            {
                if (m_hp->m_debug_on)
                    cout << "no splitting function found" << endl << flush;

                // Finalize the leaf node and clean up some stuff
                // NOTE: for ARFs, we are not allowed to re-estimate the final prediction with the
				// target values, because these are only the residuals at this point, because we could
				// also have made a split node!!!
                //if (this->m_is_ADFTree && this->m_prediction_type_ADF == 1)
                //	cnode->MakeFinalLeaf(0);
                //else
                //	cnode->MakeFinalLeaf();

                // ==> 2014-03-07 <==
                // No need for this separation! We never re-calculate any statistics in a final leaf node,
                // because we always compute the full statistics (class + regr) in any intermediate node.
                cnode->MakeFinalLeaf();

                // delete the split function if no split was found
                delete(spf);
            }
            else
            {
                if (m_hp->m_debug_on)
                    cout << "we found a splitting function for node " << cnode_id << endl << flush;

                // ... and split the samples
                DataSet<Sample, Label> leftChildDataset, rightChildDataset;
                this->SplitNode(spf, cnode_dataset, leftChildDataset, rightChildDataset);
                if (m_hp->m_debug_on)
                {
                    cout << "Data split:" << endl;
                    cout << "Left child gets " << leftChildDataset.size() << " samples" << endl;
                    cout << "Right child gets " << rightChildDataset.size() << " samples" << endl;
                }

                // create child nodes
                // LEFT
                int left_child_id = m_num_nodes++;
                Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* childnode_left = new Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>(m_hp, m_appcontext, left_child_id, cnode_depth+1, NODE_TYPE::INTERMEDIATE_LEAF, leftChildDataset);
                m_nodes.push_back(childnode_left);
                m_pq.push_back(childnode_left);
                if (m_hp->m_debug_on)
                    cout << "Left child created" << endl << flush;

                // RIGHT
                int right_child_id = m_num_nodes++;
                Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* childnode_right = new Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>(m_hp, m_appcontext, right_child_id, cnode_depth+1, NODE_TYPE::INTERMEDIATE_LEAF, rightChildDataset);
                m_nodes.push_back(childnode_right);
                m_pq.push_back(childnode_right);
                if (m_hp->m_debug_on)
                    cout << "Right child created" << endl << flush;


                // INFO: for the case of regression and ADF (ie, ARF), we have to add the
				// prediction from the leaf to the target values of the two new child nodes!!!
				// -> additive stage-wise classifier!!!
                if (this->m_is_ADFTree && this->m_prediction_type_ADF == 1)
                {
                	// The following only applies for a joint classification-regression task (e.g., Hough Forests)
                	// CAUTION: in our old implementation, there was an error: Assume we have a node with
                	// a few positive and a few negative samples and make a split such that all one
                	// node has only negative samples. Then, we can't make a regression prediction for the
                	// all negative node! -> in our old implemenation, the vote was simply set to a zero-vector.
                	// However, when taking the mean later for calculating the residuals and computing the
                	// global loss, this distorts the result!!!
                	// What shall we do then? In general, what shall we do, if a sample falls in a negative
                	// node and we want to calculate the forest regression prediction?
                	// In the old implementation, the class probability was completely ignored, and we only
                	// concentrated on the regression part.
                	childnode_left->m_leafstats->AddTarget(cnode->m_leafstats);
                	childnode_right->m_leafstats->AddTarget(cnode->m_leafstats);
                }


                // sort the priority queue
                sort(m_pq.begin(), m_pq.end(), NodeCompare<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>());

                // update node statistics and counters
                m_num_leafs += 2;
                m_nodes_in_depth[cnode_depth+1] += 2;

                // remove the current (intermediate) leaf node ... (only if we do NOT accumulate the confidences)
                m_num_leafs--;
                cnode->MakeSplitNode(spf);
                // ... and replace it with a split node.
                m_treetable(cnode_id, 1) = left_child_id;
                m_treetable(cnode_id, 2) = right_child_id;

                // resize the matrix and add two new child nodes
                MatrixXd tmpTable = MatrixXd::Zero(m_treetable.rows()+2, m_split_store_size);
                for (int r = 0; r < m_treetable.rows(); r++)
                    for (int c = 0; c < m_treetable.cols(); c++)
                        tmpTable(r, c) = m_treetable(r, c);
                this->m_treetable = tmpTable;
                this->m_treetable(left_child_id, 0) = cnode_depth + 1;
                this->m_treetable(right_child_id, 0) = cnode_depth + 1;
            }
        }
        else
        {
            if (m_hp->m_debug_on)
                cout << "Do not split node " << cnode_id << " due to general restrictions (depth, pureness, ...)" << endl;

            // Finalize the leaf node and clean up some stuff
            // NOTE: for ARFs, we are not allowed to re-estimate the final prediction with the
            // target values, because these are only the residuals at this point, because we could
            // also have made a split node!!!
            //if (this->m_is_ADFTree && this->m_prediction_type_ADF == 1)
			//	cnode->MakeFinalLeaf(0);
			//else
			//	cnode->MakeFinalLeaf();

            // ==> 2014-03-07 <==
			// No need for this separation! We never re-calculate any statistics in a final leaf node,
			// because we always compute the full statistics (class + regr) in any intermediate node.
            cnode->MakeFinalLeaf();
        }
        // delete the split-evaluator
        delete(spe);
    }
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
bool
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::FindSplittingFunction(SplitFunction* spf, SplitEvaluator* spe, DataSet<Sample, Label>& dataset_parent, int depth, int node_id)
{
	// Initialize the splitfunction to some default value in case that no hypothesis is found
    spf->SetRandomValues();
    spf->SetThreshold(1.0);

    // make a temp splitfunction
    SplitFunction* tmpspf = new SplitFunction(m_appcontext);

    // Initial declarations
    double bestSplitScore = 1e16;
    vector< std::pair<double, int> > responses;
    pair<double, double> cur_score_and_threshold = make_pair(0.0, 0.0);
    bool found_split = false;


    // Start searching for hypothesis
    for (int i = 0; i < m_hp->m_num_node_tests; i++)
    {
    	// choose a new random test
        tmpspf->SetRandomValues();

        // calculate responses
        responses.clear();
		responses.resize(dataset_parent.size());
		for (size_t s = 0; s < dataset_parent.size(); s++)
		{
			responses[s].first = tmpspf->CalculateResponse(dataset_parent[s]->m_sample);
			responses[s].second = (int)s;
		}

        // Sort the responses
        sort(responses.begin(), responses.end());

        // find best threshold and quality measure for the current split
        bool found_th = spe->CalculateScoreAndThreshold(dataset_parent, responses, cur_score_and_threshold);

        // min search
        if (found_th && cur_score_and_threshold.first < bestSplitScore)
        {
        	found_split = true;
            bestSplitScore = cur_score_and_threshold.first;
            spf->SetSplit(tmpspf);
            spf->SetThreshold(cur_score_and_threshold.second);
        }
    }

    delete(tmpspf);

    return found_split;
}



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::SplitNode(SplitFunction* spf, DataSet<Sample, Label>& dataset_parent, DataSet<Sample, Label>& dataset_leftChild, DataSet<Sample, Label>& dataset_rightChild)
{
    int num_samples = (int)dataset_parent.size();
    for (unsigned int s = 0; s < num_samples; s++)
    {
        if (spf->Split(dataset_parent[s]->m_sample) == 0)
        	dataset_leftChild.AddLabelledSample(dataset_parent[s]);
        else
        	dataset_rightChild.AddLabelledSample(dataset_parent[s]);
    }
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::GetRandomSampleSubsetForSplitting(DataSet<Sample, Label>& dataset_full, DataSet<Sample, Label>& dataset_subsample, int num_samples)
{
	// GENERIC VERSION as we do not have access to class labes, it might be any label space (e.g., regression)
	// number of samples really used
	int num_samples_used = min(num_samples, (int)dataset_full.size());

	// make a new dataset with pre-allocated memory!
	dataset_subsample.Clear();
	dataset_subsample.Resize(num_samples_used);

	// find a random subset
	vector<int> randinds = randPermSTL((int)dataset_full.size());
	for (int i = 0; i < dataset_subsample.size(); i++)
	{
		dataset_subsample.SetLabelledSample(i, dataset_full[randinds[i]]);
	}
}



#endif /* RANDOMTREE_CPP_ */

