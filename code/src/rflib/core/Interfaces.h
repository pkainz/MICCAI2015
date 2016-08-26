/*
 * Interfaces.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */


// TODO: this should actually be kept up-to-date ;)

#ifndef INTERFACES_H_
#define INTERFACES_H_


#include "DataSet.h"
#include "LabelledSample.h"
#include <fstream>




class ISample
{
	// constructors/destructors
	ISample() {}
	virtual ~ISample() {}

	// I/O methods
	virtual void Save(std::ofstream& out);
	virtual void Load(std::ifstream& in);
};


class ILabel
{
public:
	// constructors/destructors
	ILabel() {}
	virtual ~ILabel() {}

	// I/O stuff
	virtual void Save(std::ofstream& out);
	virtual void Load(std::ifstream& in);

};


template<typename Sample, typename Label, typename AppContext>
class ILeafNodeStatistics
{
public:
	// constructors/destructors
	ILeafNodeStatistics(AppContext* appcontextin);
	virtual ~ILeafNodeStatistics() {}

	// data related methods
	virtual void Aggregate(DataSet<Sample, Label> dataset, int full) = 0;
	virtual void Aggregate(ILeafNodeStatistics<Sample, Label, AppContext>* leafstatsin) = 0;
	virtual void UpdateStatistics(LabelledSample<Sample, Label>* labelled_sample) = 0;
	static ILeafNodeStatistics Average(std::vector<ILeafNodeStatistics> leafstats, AppContext* apphp);

	// ADF specific methods
	virtual std::vector<double> CalculateADFTargetResidual(Label gt_label, int prediction_type) = 0;

	// I/O methods
	virtual void Save(std::ofstream& out) = 0;
	virtual void Load(std::ifstream& in) = 0;
};



template<typename Sample, typename AppContext>
class ISplitFunction
{
public:
	// constructurs/destructors
	ISplitFunction(AppContext* appcontextin);
	virtual ~ISplitFunction() {}

	// split methods
	virtual void SetRandomValues() = 0;
	virtual void SetThreshold(double th) = 0;
	virtual void SetSplit(ISplitFunction* spf) = 0; // COPY THE DATA from spf to the current split!
	virtual int Split(Sample& sample) = 0;
	virtual double CalculateResponse(Sample& sample) = 0;

	// I/O methods
	virtual void Save(std::ofstream& out) = 0;
	virtual void Load(std::ifstream& in) = 0;
};



template<typename Sample, typename Label, typename AppContext>
class ISplitEvaluator
{
	// constructors/destructors
	ISplitEvaluator(AppContext* appcontextin, int depth, DataSet<Sample, Label> dataset);
	virtual ~ISplitEvaluator() {}

	// evaluation methods
	virtual bool DoFurtherSplitting(DataSet<Sample, Label> dataset, int depth);
	virtual bool CalculateScoreAndThreshold(DataSet<Sample, Label> dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold) = 0;
};



#endif /* INTERFACES_H_ */
