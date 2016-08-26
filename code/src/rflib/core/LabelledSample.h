/*
 * DataSample.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LABELLEDSAMPLE_H_
#define LABELLEDSAMPLE_H_

#include <fstream>


//namespace
//{

template<typename Sample, typename Label>
class LabelledSample
{
public:

	/**
	 * Constructor for a LabelledSample. It should take a Sample, a Label, a weight and a global index.
	 * CAUTION: the given weight here is irrelevant as the weight for a sample has moved to the Label class.
	 *
	 * @param[in] insample features
	 * @param[in] inlabel label
	 * @param[in] inweight for the sample [depricated???]
	 * @param[in] inglobalindex an index of that sample in a global context, e.g., a complete data set
	 */
    explicit LabelledSample(Sample insample, Label inlabel, double inweight, int inglobalindex) :
    		m_sample(insample), m_label(inlabel), m_weight(inweight), m_global_index(inglobalindex)
    {
    }

    ~LabelledSample() { }

    void Save(std::ofstream& out);
    void Load(std::ifstream& in);

    // =======================================================
    // members
    Sample m_sample;
    Label m_label;
    double m_weight;
    int m_global_index;
};

//}


#include "LabelledSample.cpp"

#endif /* LABELLEDSAMPLE_H_ */
