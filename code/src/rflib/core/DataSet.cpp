/*
 * DataSet.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef DATASET_CPP_
#define DATASET_CPP_


#include "DataSet.h"



template<typename Sample, typename Label>
DataSet<Sample, Label>::DataSet()
{
}


template<typename Sample, typename Label>
DataSet<Sample, Label>::DataSet(int num_samples_to_allocate)
{
	this->m_samples.resize(num_samples_to_allocate);
}


template<typename Sample, typename Label>
DataSet<Sample, Label>::~DataSet()
{
}


template<typename Sample, typename Label>
void DataSet<Sample, Label>::AddSample(Sample insample, Label inlabel, double inweight, int inglobal_index)
{
	m_samples.push_back(new LabelledSample<Sample, Label>(insample, inlabel, inweight, inglobal_index));
}


template<typename Sample, typename Label>
void DataSet<Sample, Label>::AddLabelledSample(LabelledSample<Sample, Label>* insample)
{
	m_samples.push_back(insample);
}


template<typename Sample, typename Label>
void DataSet<Sample, Label>::RemoveSample(unsigned int idx)
{
	delete(m_samples[idx]);
	m_samples.erease(idx);
}


template<typename Sample, typename Label>
LabelledSample<Sample, Label>* DataSet<Sample, Label>::GetLabelledSample(unsigned int idx) const
{
	return m_samples[idx];
}


template<typename Sample, typename Label>
void DataSet<Sample, Label>::SetLabelledSample(unsigned int idx, LabelledSample<Sample, Label>* insample)
{
	this->m_samples[idx] = insample;
}


template<typename Sample, typename Label>
void DataSet<Sample, Label>::GetDatasetProperties(int& out_num_samples) const
{
    out_num_samples = (int)this->m_samples.size();
}


template<typename Sample, typename Label>
DataSet<Sample, Label>* DataSet<Sample, Label>::SplitData(double split_ratio)
{
	int num_samples_new = (int)round((double)this->m_samples.size() * split_ratio);

    // define a random permutation
	std::vector<int> rand_inds = randPermSTL(this->m_samples.size());

    // Do the split
	DataSet<Sample, Label> new_dataset;
    for (int i = 0; i < this->m_samples.size(); i++)
    {
    	// move this sample to the new dataset
        if (i < num_samples_new)
        {
        	new_dataset.AddLabelledSample(this->m_samples[rand_inds[i]]);
        	this->m_samples.erase(rand_inds[i]);
        }
    }

    return new_dataset;
}

template<typename Sample, typename Label>
size_t DataSet<Sample, Label>::size() const
{
	return m_samples.size();
}

template<typename Sample, typename Label>
void DataSet<Sample, Label>::Clear()
{
	this->m_samples.clear();
}

template<typename Sample, typename Label>
void DataSet<Sample, Label>::Resize(unsigned int new_size)
{
	this->Clear();
	this->m_samples.resize((std::size_t)new_size);
}

template<typename Sample, typename Label>
void DataSet<Sample, Label>::DeleteAllSamples()
{
	for (unsigned int i = 0; i < m_samples.size(); i++)
		delete(m_samples[i]);
	this->Clear();
}

template<typename Sample, typename Label>
void DataSet<Sample, Label>::Save(std::string savepath)
{
	std::ofstream out(savepath.c_str(), std::ios::binary);
	out << (int)this->m_samples.size() << std::endl;
	for (int i = 0; i < (int)m_samples.size(); i++)
		m_samples[i]->Save(out);
	out.flush();
	out.close();
}

template<typename Sample, typename Label>
void DataSet<Sample, Label>::Load(std::string loadpath)
{
	std::ifstream in(loadpath.c_str(), std::ios::in);
	int num_samples;
	in >> num_samples;
	m_samples.resize(num_samples);
	for (int i = 0; i < (int)m_samples.size(); i++)
	{
		m_samples[i] = new LabelledSample<Sample, Label>();
		m_samples[i]->Load(in);
	}
	in.close();
}


template<typename Sample, typename Label>
LabelledSample<Sample, Label>* DataSet<Sample, Label>::operator [](unsigned int idx) const
{
	return this->GetLabelledSample(idx);
}


#endif /* DATASET_CPP_ */
