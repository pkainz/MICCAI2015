/*
 * LeafNodeStatisticsMLClass.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LEAFNODESTATISTICSMLCLASS_H_
#define LEAFNODESTATISTICSMLCLASS_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include <fstream>

#include "Interfaces.h"
#include "LabelledSample.h"
#include "LabelMLClass.h"
#include "AppContext.h"
#include "DataSet.h"

using namespace std;
using namespace Eigen;


template<typename Sample, typename TAppContext>
class LeafNodeStatisticsMLClass
{
public:

	// Constructors & destructors
	LeafNodeStatisticsMLClass(TAppContext* appcontextin);
	virtual ~LeafNodeStatisticsMLClass();

	// data methods
	virtual void Aggregate(DataSet<Sample, LabelMLClass>& dataset, int is_final_leaf);
	virtual void Aggregate(LeafNodeStatisticsMLClass* leafstatsin);
	virtual void UpdateStatistics(LabelledSample<Sample, LabelMLClass>* labelled_sample);
	static LeafNodeStatisticsMLClass Average(std::vector<LeafNodeStatisticsMLClass*> leafstats, TAppContext* apphp);
	virtual void DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std);

	// ADF specific stuff (actually only for ARF)
	virtual void AddTarget(LeafNodeStatisticsMLClass* leafnodestats);
	virtual std::vector<double> CalculateADFTargetResidual(LabelMLClass gt_label, int prediction_type);

    // analysis method
    virtual void Print();

	virtual void Save(std::ofstream& out);
	virtual void Load(std::ifstream& in);


	// public memberes
	int m_num_samples;
	std::vector<double> m_class_histogram;

protected:

	// protected members
    TAppContext* m_appcontext;

};

#include "LeafNodeStatisticsMLClass.cpp"


#endif /* LEAFNODESTATISTICSML_H_ */
