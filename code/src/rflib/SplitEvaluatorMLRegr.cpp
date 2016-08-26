/*
 * SplitEvaluatorMLRegr.cpp
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITEVALUATORMLREGR_CPP_
#define SPLITEVALUATORMLREGR_CPP_

#include "SplitEvaluatorMLRegr.h"


template<typename Sample, typename TAppContext>
SplitEvaluatorMLRegr<Sample, TAppContext>::SplitEvaluatorMLRegr(TAppContext* appcontextin, int depth, DataSet<Sample, LabelMLRegr>& dataset) : m_appcontext(appcontextin)
{
}


template<typename Sample, typename TAppContext>
SplitEvaluatorMLRegr<Sample, TAppContext>::~SplitEvaluatorMLRegr()
{
}


template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLRegr<Sample, TAppContext>::DoFurtherSplitting(DataSet<Sample, LabelMLRegr>& dataset, int depth)
{
	if (depth >= (this->m_appcontext->max_tree_depth-1) || (int)dataset.size() < this->m_appcontext->min_split_samples)
		return false;
	return true;
}


template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLRegr<Sample, TAppContext>::CalculateScoreAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
    if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
		throw std::logic_error("SplitEvaluatorMLRegr: ordinal splits not implemented for regression evaluator!");

	switch (m_appcontext->splitevaluation_type_regression)
	{
	case SPLITEVALUATION_TYPE_REGRESSION::REDUCTION_IN_VARIANCE:
        return CalculateOffsetCompactnessAndThresholdOnline(dataset, responses, score_and_threshold);
		break;
	case SPLITEVALUATION_TYPE_REGRESSION::DIFF_ENTROPY_GAUSS:
		return CalculateMVNPluginAndThreshold(dataset, responses, score_and_threshold);
		break;
	case 20: // Diagonal Approx. of MVN-Plugin
		// TODO: this could be implemented, use the method from above and make a switch:
		// 		 it's just an element-wise multiplication with the identity matrix!
		throw std::logic_error("SplitEvaluatorMLRegr::Not implemented regression loss type");
		return false;
		break;
	case 21: // MultiVariateNormal-UniformMinVarianceUnbiasedEstimate, see [Nowozin, ICML'12]
		throw std::logic_error("SplitEvaluatorMLRegr::Not implemented regression loss type");
		return false;
		break;
	case 22: // 1-NearestNeighbor loss, see [Nowozin, ICML'12]
		throw std::logic_error("SplitEvaluatorMLRegr::Not implemented regression loss type");
		return false;
		break;
	default:
		throw std::logic_error("SplitEvaluatorMLRegr::splitfunction_regression_loss_type not defined");
		return false;
		break;
	}
}












// PROTECTED/HELPER METHODS
template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLRegr<Sample, TAppContext>::CalculateMVNPluginAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double,double>& score_and_threshold)
{
	// In: samples, sorted responses, out: optimality-measure + threshold

	// Initialize the variables and counters
	double InfoGain, LEntropy, REntropy, bestThreshold = 0.0, BestInfoGain = 1e16;
	double LTotal = 0.0, RTotal = 0.0, LSqNormTotal = 0.0, RSqNormTotal = 0.0;
	VectorXd RMean = VectorXd::Zero(m_appcontext->num_target_variables);
	VectorXd LMean = VectorXd::Zero(m_appcontext->num_target_variables);
	VectorXd RSum = VectorXd::Zero(m_appcontext->num_target_variables);
	VectorXd LSum = VectorXd::Zero(m_appcontext->num_target_variables);
	MatrixXd LCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
	MatrixXd RCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
	vector<int> RSamples, LSamples;
	bool found = false;

	// Calculate random thresholds and sort them
	double min_response = responses[0].first;
	double max_response = responses[responses.size()-1].first;
	double d = (max_response - min_response);
	vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
	for (int i = 0; i < random_thresholds.size(); i++)
		random_thresholds[i] = (randDouble() * d) + min_response;
	sort(random_thresholds.begin(), random_thresholds.end());

	// First, put everything in the right node
	RSamples.resize(responses.size());
	for (int r = 0; r < responses.size(); r++)
	{
		double csw = dataset[responses[r].second]->m_weight;
		Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;
		RSum += csw * cst;
		RTotal += csw;
		RSamples[r] = responses[r].second;
	}
	RMean = RSum / RTotal;

	// Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
	int th_idx = 0;
	bool stop_search = false;
	for (int r = 0; r < responses.size(); r++)
	{
		// if the current sample is smaller than the current threshold put it to the left side
		if (responses[r].first <= random_thresholds[th_idx])
		{
			// move the current response from the right node to the left node
			double csw = dataset[responses[r].second]->m_weight;
			Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;
			RSum -= csw * cst;
			RTotal -= csw;
			if (RTotal < 0.0) // should never happen
				RTotal = 0.0;
			LSum += csw * cst;
			LTotal += csw;
			LSamples.push_back(RSamples[0]);
			RSamples.erase(RSamples.begin());
		}
		else
		{
			if (LTotal > 0.0 && RTotal > 0.0)
			{
				// now, we have to check the split quality, this would be a valid split

				// RIGHT: Weighted mean
				RMean = RSum / RTotal;
				// weighted co-variance [http://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance#Weighted_samples]
				RCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
				RSqNormTotal = 0.0;
				for (int s = 0; s < RSamples.size(); s++)
				{
					Eigen::VectorXd cst = dataset[RSamples[s]]->m_label.regr_target;
					RCov += dataset[RSamples[s]]->m_weight * ((cst - RMean) * (cst - RMean).transpose());
					RSqNormTotal += pow(dataset[RSamples[s]]->m_weight/RTotal, 2.0);
				}
				RCov /= RTotal; // this normalization is important: sum_i w_i = 1 should hold for the weighted covariance
				if (RSqNormTotal < 1.0) // this happens if only one sample is available!
					RCov /= (1.0 - RSqNormTotal);
				double RCovDet = RCov.determinant();
				if (RCovDet <= 0.0) // happens if 2 samples only available -> one eigval=0 -> minimal quantization errors -> -5e-11
					RCovDet = 1e-10;
				//REntropy = (double)m_num_target_variables/2.0 - (double)m_num_target_variables/2.0 * log(2.0 * M_PI) + 0.5 * log(RCovDet);
				REntropy = log(RCovDet);
				if (REntropy <= 0.0)
					REntropy = 0.0;

				// LEFT: Weighted mean
				LMean = LSum / LTotal;
				// weighted co-variance
				LCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
				LSqNormTotal = 0.0;
				for (int s = 0; s < LSamples.size(); s++)
				{
					Eigen::VectorXd cst = dataset[LSamples[s]]->m_label.regr_target;
					LCov += dataset[LSamples[s]]->m_weight * ((cst - LMean) * (cst - LMean).transpose());
					LSqNormTotal += pow(dataset[LSamples[s]]->m_weight/LTotal, 2.0);
				}
				if (LSamples.size() == 0)
				{
					cout << LCov << endl;
					cout << LSqNormTotal << endl;
				}
				LCov /= LTotal;
				if (LSqNormTotal < 1.0) // this happens if only one sample is available!
					LCov /= (1.0 - LSqNormTotal);
				double LCovDet = LCov.determinant();
				if (LCovDet <= 0.0) // happens if 2 samples only available -> one eigval=0 -> minimal quantization errors -> -5e-11
					LCovDet = 1e-10;
				//LEntropy = (double)m_num_target_variables/2.0 - (double)m_num_target_variables/2.0 * log(2.0 * M_PI) + 0.5 * log(LCovDet);
				LEntropy = log(LCovDet);
				if (LEntropy <= 0.0)
					LEntropy = 0.0;

				// combine left and right entropy measures (weighted!!!)
				InfoGain = (LTotal*LEntropy + RTotal*REntropy) / (LTotal + RTotal);

				if (this->m_appcontext->debug_on)
					cout << "Eval: " << InfoGain << ", LTotal=" << LTotal << ", RTotal=" << RTotal << "(" << LEntropy << ", " << REntropy << ")" << endl;

				if (InfoGain < BestInfoGain)
				{
					BestInfoGain = InfoGain;
					bestThreshold = random_thresholds[th_idx];
					found = true;
				}
			}

			// next, we have to find the next random threshold that is larger than the current response
			// -> there might be several threshold within the gap between the last response and this one.
			while (responses[r].first > random_thresholds[th_idx])
			{
				if (th_idx < (random_thresholds.size()-1))
				{
					th_idx++;
					// CAUTION::: THIS HAS TO BE INCLUDED !!!!!!!!!!!??????
					r--; // THIS IS IMPORTANT, WE HAVE TO CHECK THE CURRENT SAMPLE AGAIN!!!
				}
				else
				{
					stop_search = true;
					break; // all thresholds tested
				}
			}
			// now, we can go on with the next response ...
		}
		if (stop_search)
			break;
	}

	score_and_threshold.first = BestInfoGain;
	score_and_threshold.second = bestThreshold;
	return found;
}

template<typename Sample, typename TAppContext>
bool
SplitEvaluatorMLRegr<Sample, TAppContext>::CalculateOffsetCompactnessAndThresholdOnline(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
    // INFO: this is only valid for the Reduction-In-Variance!
    // In: samples, sorted responses, out:offset_measure+threshold

    // PK: CELL DETECTION REGRESSION --> FIX lblIdx = 0, since we just have one class!
    int lblIdx = 0;

    // Initialize the counters
    double curr_variance, LTotal = 0.0, RTotal = 0.0, bestThreshold = 0.0, best_variance = 1e16;
    vector<VectorXd> LMean_class(this->m_appcontext->num_classes, Eigen::VectorXd::Zero(this->m_appcontext->num_target_variables));
    vector<VectorXd> RMean_class(this->m_appcontext->num_classes, Eigen::VectorXd::Zero(this->m_appcontext->num_target_variables));
    vector<double> LVarSq(this->m_appcontext->num_classes, 0.0), RVarSq(this->m_appcontext->num_classes, 0.0); // left variance squared (for single class problem remove the std::vector!)
    vector<double> LTotal_class(this->m_appcontext->num_classes), RTotal_class(this->m_appcontext->num_classes); // for sincle class problems remove the outer std::vector!
    bool found = false;

    // Calculate random thresholds and sort them
    double min_response = responses[0].first;
    double max_response = responses[responses.size()-1].first;
    double d = (max_response - min_response);
    vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
    for (int i = 0; i < random_thresholds.size(); i++)
    {
        random_thresholds[i] = (randDouble() * d) + min_response;
    }
    sort(random_thresholds.begin(), random_thresholds.end());

    // First, put everything in the right node
    for (int r = 0; r < responses.size(); r++)
    {
//      int lblIdx = dataset[responses[r].second]->m_label.class_label;

//		// if this sample is negative, skip it !
//		if (dataset[responses[r].second]->m_label.vote_allowed == false)
//			continue;
//		//RSamples[lblIdx].push_back(responses[r].second);

        // 4 online
        double csw = dataset[responses[r].second]->m_label.regr_weight; // current sample weight
        Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target; // current sample target

        double temp = RTotal_class[lblIdx] + csw;
        //VectorXd delta = cst - RMean;
        VectorXd delta = cst - RMean_class[lblIdx];
        VectorXd R = delta * csw / temp;
        //RMean += R;
        RMean_class[lblIdx] += R;
        RVarSq[lblIdx] = RVarSq[lblIdx] + RTotal_class[lblIdx] * delta.dot(delta) * csw / temp;
        RTotal_class[lblIdx] = temp;
        //RVar = RVarSq/RTotal;
    }

    // Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
    int th_idx = 0;
    bool stop_search = false;
    for (int r = 0; r < responses.size(); r++)
    {
//        int lblIdx = dataset[responses[r].second]->m_label.class_label;

//		// if this sample is negative, skip it !
//		if (dataset[responses[r].second]->m_label.vote_allowed == false)
//			continue;

        // if the current sample is smaller than the current threshold put it to the left side
        if (responses[r].first <= random_thresholds[th_idx])
        {
            // Remove the current sample from the right side and put it on the left side ...
            //LSamples[lblIdx].push_back(RSamples[lblIdx][0]);
            //RSamples[lblIdx].erase(RSamples[lblIdx].begin());

            // 4 online
            double csw = dataset[responses[r].second]->m_label.regr_weight;
            Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;

            double temp = RTotal_class[lblIdx] - csw;
            //VectorXd delta = cst - RMean;
            VectorXd delta = cst - RMean_class[lblIdx];
            VectorXd R = delta * csw / temp;
            //RMean -= R;
            RMean_class[lblIdx] -= R;
            RVarSq[lblIdx] = RVarSq[lblIdx] - RTotal_class[lblIdx] * delta.dot(delta) * csw / temp;
            RTotal_class[lblIdx] = temp;
            //RVar = RVarSq/RTotal;

            temp = LTotal_class[lblIdx] + csw;
            //delta = cst - LMean;
            delta = cst - LMean_class[lblIdx];
            R = delta * csw / temp;
            //LMean += R;
            LMean_class[lblIdx] += R;
            LVarSq[lblIdx] = LVarSq[lblIdx] + LTotal_class[lblIdx] * delta.dot(delta) * csw / temp;
            LTotal_class[lblIdx] = temp;
            //LVar = LVarSq/LTotal;
        }
        else
        {
            // ok, now we found the first sample having higher response than the current threshold
            //double curr_variance_old = EvaluateRegressionLoss(dataset, LSamples, RSamples, LTotal, RTotal);
            //curr_variance = LTotal / (LTotal+RTotal) * LVar + RTotal / (LTotal+RTotal) * RVar;
            // as we see from this formula: LTotal/xx * LVarUnnorm/LTotal + ...
            // we can drop the LTotal normalization for the LVar & RVar
            curr_variance = 0.0;
            LTotal = 0.0, RTotal = 0.0;
            for (int c = 0; c < m_appcontext->num_classes; c++)
            {
                double var_class = (LVarSq[c] + RVarSq[c]) / (LTotal_class[c] + RTotal_class[c]);
                if (LTotal_class[c] == 0.0)
                    var_class = RVarSq[c] / RTotal_class[c];
                if (RTotal_class[c] == 0.0)
                    var_class = LVarSq[c] / LTotal_class[c];
                if (LTotal_class[c] == 0.0 && RTotal_class[c] == 0.0)
                    var_class = 0.0;
                curr_variance += var_class;
                LTotal += LTotal_class[c];
                RTotal += RTotal_class[c];
            }
            curr_variance /= (double)m_appcontext->num_classes;

            if (curr_variance < best_variance && LTotal > 0.0 && RTotal > 0.0)
            {
                best_variance = curr_variance;
                bestThreshold = random_thresholds[th_idx];
                found = true;
            }

            // next, we have to find the next random threshold that is larger than the current response
            // -> there might be several threshold within the gap between the last response and this one.
            while (responses[r].first > random_thresholds[th_idx])
            {
                if (th_idx < (random_thresholds.size()-1))
                {
                    th_idx++;
                    r--; // THIS IS IMPORTANT, WE HAVE TO CHECK THE CURRENT SAMPLE AGAIN!!!
                }
                else
                {
                    stop_search = true;
                    break; // all thresholds tested
                }
            }
            // now, we can go on with the next response ...
        }

        if (stop_search)
            break;
    }

    score_and_threshold.first = best_variance;
    score_and_threshold.second = bestThreshold;
    return found;
}


#endif /* SPLITEVALUATORMLREGR_CPP_ */
