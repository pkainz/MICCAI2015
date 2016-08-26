/*
 * LeafNodeStatisticsMLRegr.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LEAFNODESTATISTICSMLREGR_H_
#define LEAFNODESTATISTICSMLREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include <fstream>

#include "Interfaces.h"
#include "LabelledSample.h"
#include "LabelMLRegr.h"
#include "DataSet.h"

using namespace std;
using namespace Eigen;


template<typename Sample, typename TAppContext>
class LeafNodeStatisticsMLRegr
{
public:

	// Constructors & destructors
    LeafNodeStatisticsMLRegr(TAppContext* appcontextin);
	virtual ~LeafNodeStatisticsMLRegr();

	// data methods
    virtual void Aggregate(DataSet<Sample, LabelMLRegr>& dataset, int is_final_leaf);
	virtual void Aggregate(LeafNodeStatisticsMLRegr* leafstatsin);
    virtual void UpdateStatistics(LabelledSample<Sample, LabelMLRegr>* labelled_sample);
    static LeafNodeStatisticsMLRegr Average(std::vector<LeafNodeStatisticsMLRegr*> leafstats, TAppContext* apphp);
	virtual void DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std);

	// ADF-specific
	virtual void AddTarget(LeafNodeStatisticsMLRegr* leafnodestats);
	virtual std::vector<double> CalculateADFTargetResidual(LabelMLRegr gt_label, int prediction_type);

    // analysis method
    virtual void Print();

	// I/O methods
	virtual void Save(std::ofstream& out);
	virtual void Load(std::ifstream& in);


	// public memberes
	int m_num_samples;
	Eigen::VectorXd m_prediction;

protected:

	// protected members
    TAppContext* m_appcontext;

};

#include "LeafNodeStatisticsMLRegr.cpp"

#endif /* LEAFNODESTATISTICSML_H_ */
