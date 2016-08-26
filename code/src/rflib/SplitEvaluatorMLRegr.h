/*
 * SplitEvaluatorMLRegr.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITEVALUATORMLREGR_H_
#define SPLITEVALUATORMLREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include "LabelMLRegr.h"
#include "AppContext.h"

#include "icgrf.h"

using namespace std;
using namespace Eigen;


template<typename Sample, typename TAppContext>
class SplitEvaluatorMLRegr
{
public:

	// Constructors & destructors
    SplitEvaluatorMLRegr(TAppContext* appcontextin, int depth, DataSet<Sample, LabelMLRegr>& dataset);
    virtual ~SplitEvaluatorMLRegr();

    bool DoFurtherSplitting(DataSet<Sample, LabelMLRegr>& dataset, int depth);
    bool CalculateScoreAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);

protected:

    // Regression stuff
    bool CalculateMVNPluginAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double,double>& score_and_threshold);
    bool CalculateOffsetCompactnessAndThresholdOnline(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);

	// members
    TAppContext* m_appcontext;

};

#include "SplitEvaluatorMLRegr.cpp"

#endif /* SPLITFUNCTIONSML_H_ */
