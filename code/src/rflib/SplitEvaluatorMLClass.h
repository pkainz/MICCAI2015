/*
 * SplitEvaluatorMLRegr.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITEVALUATORMLREGR_H_
#define SPLITEVALUATORMLREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include "AppContext.h"
//#include "AppContextML.h"
#include "LabelMLClass.h"
//#include "SampleML.h"

#include "icgrf.h"

using namespace std;
using namespace Eigen;


template<typename Sample, typename TAppContext>
class SplitEvaluatorMLClass
{
public:

	// Constructors & destructors
    SplitEvaluatorMLClass(TAppContext* appcontextin, int depth, DataSet<Sample, LabelMLClass>& dataset);
    virtual ~SplitEvaluatorMLClass();

    bool DoFurtherSplitting(DataSet<Sample, LabelMLClass>& dataset, int depth);
    bool CalculateScoreAndThreshold(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);

protected:

    // Classification stuff
	bool CalculateEntropyAndThreshold(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold, int use_gini);
	bool CalculateEntropyAndThresholdOrdinal(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold, int use_gini);
	bool CalculateSpecificLossAndThreshold(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);
	double ComputeLoss(vector<double> p, int c, ADF_LOSS_CLASSIFICATION::Enum wut);

	// members
	TAppContext* m_appcontext;

};


#include "SplitEvaluatorMLClass.cpp"







#endif /* SPLITFUNCTIONSML_H_ */
