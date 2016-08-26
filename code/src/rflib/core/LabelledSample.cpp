/*
 * LabelledSample.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LABELLEDSAMPLE_CPP_
#define LABELLEDSAMPLE_CPP_

#include "LabelledSample.h"



template<typename Sample, typename Label>
void LabelledSample<Sample, Label>::Save(std::ofstream& out)
{
	out << m_global_index << std::endl;
	out << m_weight << std::endl;
	m_sample.Save(out);
	m_label.Save(out);
}

template<typename Sample, typename Label>
void LabelledSample<Sample, Label>::Load(std::ifstream& in)
{
	in >> m_global_index;
	in >> m_weight;
	m_sample = new Sample();
	m_sample.Load(in);
	m_label = new Label();
	m_label.Load(in);
}


#endif /* LABELLEDSAMPLE_CPP_ */
