/*
 * LeafNodeStatisticsMLClass.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LEAFNODESTATISTICSMLCLASS_CPP_
#define LEAFNODESTATISTICSMLCLASS_CPP_


#include "LeafNodeStatisticsMLClass.h"

template<typename Sample, typename TAppContext>
LeafNodeStatisticsMLClass<Sample, TAppContext>::LeafNodeStatisticsMLClass(TAppContext* appcontextin) : m_appcontext(appcontextin)
{
	this->m_class_histogram.resize(m_appcontext->num_classes, 1.0 / (double)m_appcontext->num_classes);
	this->m_num_samples = 0;
}


template<typename Sample, typename TAppContext>
LeafNodeStatisticsMLClass<Sample, TAppContext>::~LeafNodeStatisticsMLClass() { }



template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::Aggregate(DataSet<Sample, LabelMLClass>& dataset, int is_final_leaf)
{
	// reset the histogram
	this->m_class_histogram.clear();
	this->m_class_histogram.resize(m_appcontext->num_classes, 0.0);

	// fill the histogram
	this->m_num_samples = (int)dataset.size();
	for (size_t s = 0; s < this->m_num_samples; s++)
	{
		int labelIdx = dataset[s]->m_label.class_label;
		m_class_histogram[labelIdx] += 1.0;
	}

	// normalize it
	for (int l = 0; l < (int)m_class_histogram.size(); l++)
		m_class_histogram[l] /= (double)m_num_samples;
}


template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::Aggregate(LeafNodeStatisticsMLClass* leafstatsin)
{
	// de-normalize the histogram and
	// add the de-normalized input-leafstats
	for (size_t l = 0; l < m_class_histogram.size(); l++)
	{
		m_class_histogram[l] *= (double)m_num_samples;
		m_class_histogram[l] += (leafstatsin->m_class_histogram[l] * (double)leafstatsin->m_num_samples);
	}
	m_num_samples += leafstatsin->m_num_samples;

	// normalize the histogram
	for(size_t l = 0; l < m_class_histogram.size(); l++)
		m_class_histogram[l] /= (double)m_num_samples;
}



template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::UpdateStatistics(LabelledSample<Sample, LabelMLClass>* labelled_sample)
{
	// if no samples have been routed to this leaf, "initialize" it ...
	// this should never happen, except for the online case!
	if (m_num_samples == 0)
	{
		// use this single sample for the histogram
		for (int c = 0; c < m_class_histogram.size(); c++)
			m_class_histogram[c] = 0.0;
		m_class_histogram[labelled_sample->m_label.class_label] = 1.0;

		// increase the total number of samples
		m_num_samples++;
		return;
	}


	// Add the statics of a new sample to the histogram ...
	// denormalize the histogram
	for (size_t l = 0; l < m_class_histogram.size(); l++)
		m_class_histogram[l] *= (double)m_num_samples;

	// add the sample to the corresponding bin
	int labelIdx = labelled_sample->m_label.class_label;
	m_class_histogram[labelIdx] += 1.0;

	// increase the total number of samples
	m_num_samples++;

	// normalize the histogram again
	for (size_t l = 0; l < m_class_histogram.size(); l++)
		m_class_histogram[l] /= (double)m_num_samples;
}


template<typename Sample, typename TAppContext>
LeafNodeStatisticsMLClass<Sample, TAppContext>
LeafNodeStatisticsMLClass<Sample, TAppContext>::Average(std::vector<LeafNodeStatisticsMLClass<Sample, TAppContext>* > leafstats, TAppContext* apphp)
{
	LeafNodeStatisticsMLClass ret_stats(apphp);
	ret_stats.m_class_histogram.clear();
	ret_stats.m_class_histogram.resize(apphp->num_classes, 0.0);
	for (size_t i = 0; i < leafstats.size(); i++)
	{
		for (size_t c = 0; c < leafstats[i]->m_class_histogram.size(); c++)
			ret_stats.m_class_histogram[c] += leafstats[i]->m_class_histogram[c];
	}
	for (size_t c = 0; c < leafstats[0]->m_class_histogram.size(); c++)
		ret_stats.m_class_histogram[c] /= (double)leafstats.size();
	return ret_stats;
}


template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std)
{
	throw std::logic_error("LeafNodeStatisticsMLClass has to target variables for denormalization!");
}



template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::AddTarget(LeafNodeStatisticsMLClass* leafnodestats)
{
	// For classification (ADF) we do nothing, actually, this method isn't even called!
	throw std::runtime_error("LeafNodeStatisticsMLClass: this function should not be called in this context!");
}


template<typename Sample, typename TAppContext>
std::vector<double> LeafNodeStatisticsMLClass<Sample, TAppContext>::CalculateADFTargetResidual(LabelMLClass gt_label, int prediction_type)
{
	// prediction type can be ignored here, as it is clear that we are in a classification task

	std::vector<double> ret_vec(1);

	// OLD VERSION
	//ret_vec[0] = this->m_class_histogram[gt_label.class_label];

	// NEW VERSION
	// we now calculate the margin of the classification
	// m = p(y_gt) - max_{y != y_gt} p(y)
	double max_conf_not_gt = -1.0;
	for (int c = 0; c < this->m_class_histogram.size(); c++)
		if (c != gt_label.gt_class_label)
			if (this->m_class_histogram[c] > max_conf_not_gt)
				max_conf_not_gt = this->m_class_histogram[c];
	ret_vec[0] = this->m_class_histogram[gt_label.gt_class_label] - max_conf_not_gt;

	// return it
	return ret_vec;
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::Print()
{
    cout << this->m_num_samples << " samples; class-specific: ";
//    for (size_t c = 0; c < this->m_num_samples_class.size(); c++)
//        cout << this->m_num_samples_class[c] << " ";
//    cout << endl;
    for (size_t c = 0; c < this->m_class_histogram.size(); c++)
        cout << this->m_class_histogram[c] << " ";
    cout << endl;
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::Save(std::ofstream& out)
{
	out << m_num_samples << " ";
	out << m_class_histogram.size() << " ";
	for (int c = 0; c < m_class_histogram.size(); c++)
		out << m_class_histogram[c] << " ";
	out << endl;
}



template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLClass<Sample, TAppContext>::Load(std::ifstream& in)
{
	int num_classes;
	in >> m_num_samples >> num_classes;
	m_class_histogram.resize(num_classes);
	for (int c = 0; c < m_class_histogram.size(); c++)
		in >> m_class_histogram[c];
}



#endif /* LEAFNODESTATISTICSML_CPP_ */
