/*
 * LeafNodeStatisticsMLRegr.cpp
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LEAFNODESTATISTICSMLREGR_CPP_
#define LEAFNODESTATISTICSMLREGR_CPP_

#include "LeafNodeStatisticsMLRegr.h"


template<typename Sample, typename TAppContext>
LeafNodeStatisticsMLRegr<Sample, TAppContext>::LeafNodeStatisticsMLRegr(TAppContext* appcontextin) : m_appcontext(appcontextin)
{
	this->m_prediction = Eigen::VectorXd::Zero(m_appcontext->num_target_variables);
}

template<typename Sample, typename TAppContext>
LeafNodeStatisticsMLRegr<Sample, TAppContext>::~LeafNodeStatisticsMLRegr() {
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::Aggregate(DataSet<Sample, LabelMLRegr>& dataset, int is_final_leaf)
{
	// INFO: full is obsolete here ...

	// INFO: we always calculate the mean as leaf node statistics, though other approaches exist!

	// TODO: we could implement other statistics here, e.g., a meanshift, median, etc.

	// INFO: if it is a final leaf node, do not re-calculate the leaf nodes statistics!!!
	// Reason 1) as we always have calculate the predictions for all intermediate leaf nodes,
	//         	 there is no need for doing this recalculation, as it is already done, we just
	//           have to switch the state from Intermediate to Final.
	// Reason 2) There is also a second, more important reason: If we would recalculate the
	//           prediction at this point, ARFs won't work anymore, because we would calculate
	//           the predictions with the pseudo targets and do not add the parent anymore!
	if (!is_final_leaf)
	{
		// reset the target prediction
		this->m_prediction = VectorXd::Zero(m_appcontext->num_target_variables);

		// fill the prediction
		this->m_num_samples = (int)dataset.size();
		for (int s = 0; s < this->m_num_samples; s++)
			this->m_prediction += dataset[s]->m_label.regr_target;

		// normalize it
		m_prediction /= (double)m_num_samples;
	}

}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::Aggregate(LeafNodeStatisticsMLRegr* leafstatsin)
{
	// Fuse this leafnode-statistics with the incoming one!

	// if this leaf has no samples yet, set the predictions to zero
	if (this->m_num_samples == 0)
		this->m_prediction = Eigen::VectorXd::Zero(this->m_appcontext->num_target_variables);

	// add both de-normalzed predictions
	this->m_prediction = this->m_prediction * this->m_num_samples + leafstatsin->m_prediction * leafstatsin->m_num_samples;

	// increase the number of sample in the leaf statistics
	this->m_num_samples += leafstatsin->m_num_samples;

	// normalize the new statistics
	this->m_prediction /= (double)this->m_num_samples;
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::UpdateStatistics(LabelledSample<Sample, LabelMLRegr>* labelled_sample)
{
	// if no samples have been routed to this leaf, "initialize" it ...
	// however, this should never happen!
	if (m_num_samples == 0)
	{
		this->m_prediction = labelled_sample->m_label.regr_target;
		m_num_samples = 1;
		return;
	}

	// denormalize the prediction target
	m_prediction *= (double)m_num_samples;

	// add the sample to the corresponding bin
	m_prediction += labelled_sample->m_label.regr_target;

	// increase the total number of samples
	m_num_samples++;

	// normalize the histogram again
	m_prediction /= (double)m_num_samples;
}

template<typename Sample, typename TAppContext>
LeafNodeStatisticsMLRegr<Sample, TAppContext> LeafNodeStatisticsMLRegr<Sample, TAppContext>::Average(std::vector<LeafNodeStatisticsMLRegr<Sample, TAppContext>* > leafstats, TAppContext* apphp)
{
	// add the predictions from the leafnode stats ...
    LeafNodeStatisticsMLRegr<Sample, TAppContext> ret_stats(apphp); // already initialized
    for (size_t i = 0; i < leafstats.size(); i++)
        ret_stats.m_prediction += leafstats[i]->m_prediction;

	// ... noramlize it (mean) ...
	ret_stats.m_prediction /= (double)leafstats.size();

	// ... and return the result.
	return ret_stats;
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std)
{
	throw std::logic_error("ERROR: this method is not implemented yet (LeafNodeStatisticsMLRegr.cpp:133");
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::AddTarget(LeafNodeStatisticsMLRegr* leafnodestats_src)
{
	//if (this->m_prediction.size() != leafnodestats_src->m_prediction.size())
	//	throw std::runtime_error("LeafNodeStatisticsMLRegr: AddTarget gives an error: src and dst targets have different dimensionality!");

	this->m_prediction += leafnodestats_src->m_prediction;
}

template<typename Sample, typename TAppContext>
std::vector<double> LeafNodeStatisticsMLRegr<Sample, TAppContext>::CalculateADFTargetResidual(LabelMLRegr gt_label, int prediction_type)
{
	// prediction type can be ignored here, as it is clear that we are in a regression task
	std::vector<double> ret_vec(this->m_prediction.size());

	// calculate: prediction - ground-truth
	for (size_t v = 0; v < this->m_prediction.rows(); v++)
		ret_vec[v] = this->m_prediction(v) - gt_label.regr_target_gt(v);

	// return the residual vector
	return ret_vec;
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::Print()
{
    // print the number of samples in that node
    cout << this->m_num_samples;
    cout << " samples; prediction per class (" << this->m_appcontext->num_target_variables;
    cout << " in total):";

    // print the prediction value of that node
    for (size_t c = 0; c < this->m_appcontext->num_target_variables; c++)
        cout << "[" << c << "]: " << this->m_prediction;

    cout << endl;
}


template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::Save(std::ofstream& out)
{
	out << m_num_samples << " ";
	out << m_prediction.rows() << " ";
	for (int c = 0; c < m_prediction.rows(); c++)
		out << m_prediction(c) << " ";
	out << endl;
}

template<typename Sample, typename TAppContext>
void LeafNodeStatisticsMLRegr<Sample, TAppContext>::Load(std::ifstream& in)
{
	int target_dim;
	in >> m_num_samples >> target_dim;
	m_prediction = Eigen::VectorXd::Zero(target_dim);
	for (int c = 0; c < m_prediction.rows(); c++)
		in >> m_prediction(c);
}


#endif /* LEAFNODESTATISTICSMLREGR_CPP_ */
