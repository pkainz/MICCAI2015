/*
 * DataSet.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>

#include "LabelledSample.h"
#include "RFCoreParameters.h"


//namespace icgrf
//{

template<typename Sample, typename Label>
class DataSet
{
public:

    // Constructors & Destructors
    DataSet();
    DataSet(int num_samples_to_allocate);
    ~DataSet();

    // add, remove, get samples ...
    void AddSample(Sample insample, Label inlabel, double inweight, int inglobal_index);
    void AddLabelledSample(LabelledSample<Sample, Label>* insample);
    void RemoveSample(unsigned int idx);
    LabelledSample<Sample, Label>* GetLabelledSample(unsigned int idx) const;
    void SetLabelledSample(unsigned int idx, LabelledSample<Sample, Label>* insample);

    // additional helper functions
    void GetDatasetProperties(int& out_num_samples) const;
    DataSet<Sample, Label>* SplitData(double split_ratio);
    std::size_t size() const;
    void Clear();
    void Resize(unsigned int new_size);
    void DeleteAllSamples();

    // i/o functions
    void Save(std::string savepath);
    void Load(std::string loadpath);

    // operator overloads
    LabelledSample<Sample, Label>* operator[](unsigned int idx) const;

private:

    // ===========================
    // members
    std::vector<LabelledSample<Sample, Label>* > m_samples;
};

//}

#include "DataSet.cpp"


#endif /* DATASET_H_ */
