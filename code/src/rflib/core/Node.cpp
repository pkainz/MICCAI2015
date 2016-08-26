/*
 * Node.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#include "Node.h"

#ifndef NODE_CPP_
#define NODE_CPP_



template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::Node(RFCoreParameters* hpin, AppContext* appcontextin) :
	m_hp(hpin), m_appcontext(appcontextin)
{
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::Node(RFCoreParameters* hpin, AppContext* appcontextin, int innodeid, int indepth, NODE_TYPE::Enum node_type, DataSet<Sample, Label>& dataset) :
	m_hp(hpin), m_appcontext(appcontextin), m_node_id(innodeid), m_depth(indepth),
	m_type(node_type), m_dataset(dataset), m_leafstats(NULL), m_splitfunction(NULL)
{
	if (this->m_type == NODE_TYPE::INTERMEDIATE_LEAF)
		this->MakeIntermediateLeaf();
	else if (this->m_type == NODE_TYPE::FINAL_LEAF)
		this->MakeFinalLeaf();
	else
		throw std::logic_error("Node::node constructor only works with intermediate or final leafs");
}

template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::~Node()
{
	if (this->m_is_leaf)
		delete(this->m_leafstats);
	else
		delete(this->m_splitfunction);
}



template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
int Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::Split(LabelledSample<Sample, Label>* labelled_sample_in) const
{
	//if (this->m_type == NODE_TYPE::FINAL_LEAF)
	//	throw std::logic_error("Node::trying to split a final leaf node!");

	return this->m_splitfunction->Split(labelled_sample_in->m_sample);
}



template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::MakeSplitNode(SplitFunction* spfin)
{
	this->m_is_leaf = false;
	this->m_type = NODE_TYPE::SPLIT_NODE;
	this->m_splitfunction = spfin;
	// clear the leafnodestatistics ...
	if (this->m_leafstats != NULL)
	{
		delete(this->m_leafstats);
		this->m_leafstats = NULL;
	}
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::MakeIntermediateLeaf()
{
	this->m_is_leaf = true;
	this->m_type = NODE_TYPE::INTERMEDIATE_LEAF;
	this->CalcualteLeafNodeStatistics();
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::MakeFinalLeaf()
{
	this->m_is_leaf = true;
	this->m_type = NODE_TYPE::FINAL_LEAF;
	this->CalcualteLeafNodeStatistics();
	this->m_dataset.Clear();
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::CalcualteLeafNodeStatistics(DataSet<Sample, Label>& datasetin)
{
	this->m_dataset = datasetin;
	this->CalcualteLeafNodeStatistics();
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::CalcualteLeafNodeStatistics()
{
	if (this->m_leafstats == NULL)
	{
		this->m_leafstats = new LeafNodeStatistics(m_appcontext);
	}
	if (this->m_type == NODE_TYPE::FINAL_LEAF)
		this->m_leafstats->Aggregate(this->m_dataset, 1);
	else
		this->m_leafstats->Aggregate(this->m_dataset, 0);
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::UpdateLeafnodeStatistics(LabelledSample<Sample, Label>* labelled_sample)
{
//	if (this->m_type == NODE_TYPE::SPLIT_NODE)
//		throw std::logic_error("Node: trying to update statistics on a split node");

	this->m_leafstats->UpdateStatistics(labelled_sample);
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::Save(std::ofstream& out)
{
	out << this->m_node_id << " " << this->m_depth << " " << this->m_is_leaf << std::endl;
	if (m_is_leaf)
		this->m_leafstats->Save(out);
	else
		this->m_splitfunction->Save(out);
}


template<typename Sample, typename Label, typename SplitFunction, typename LeafNodeStatistics, typename AppContext>
void Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>::Load(std::ifstream& in)
{
	in >> this->m_node_id >> this->m_depth >> this->m_is_leaf;

	if (m_is_leaf)
	{
		this->m_type = NODE_TYPE::FINAL_LEAF;
		this->m_leafstats = new LeafNodeStatistics(m_appcontext);
		this->m_leafstats->Load(in);
	}
	else
	{
		this->m_type = NODE_TYPE::SPLIT_NODE;
		this->m_splitfunction = new SplitFunction(m_appcontext);
		this->m_splitfunction->Load(in);
	}
}



#endif /* NODE_CPP_ */

