/*
 * SplitEvaluatorMLClass.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITEVALUATORMLCLASS_CPP_
#define SPLITEVALUATORMLCLASS_CPP_

#include "SplitEvaluatorMLClass.h"


template<typename Sample, typename TAppContext>
SplitEvaluatorMLClass<Sample, TAppContext>::SplitEvaluatorMLClass(TAppContext* appcontextin, int depth, DataSet<Sample, LabelMLClass>& dataset) : m_appcontext(appcontextin)
{
}


template<typename Sample, typename TAppContext>
SplitEvaluatorMLClass<Sample, TAppContext>::~SplitEvaluatorMLClass()
{
}


template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLClass<Sample, TAppContext>::DoFurtherSplitting(DataSet<Sample, LabelMLClass>& dataset, int depth)
{
	if (depth >= (this->m_appcontext->max_tree_depth-1) || (int)dataset.size() < this->m_appcontext->min_split_samples)
		return false;

	// Test pureness of the node
	int startLabel = dataset[0]->m_label.class_label;
	for (int s = 0; s < (int)dataset.size(); s++)
		if (dataset[s]->m_label.class_label != startLabel)
			return true;

	return false;
}


template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLClass<Sample, TAppContext>::CalculateScoreAndThreshold(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	switch (m_appcontext->splitevaluation_type_classification)
	{
	case SPLITEVALUATION_TYPE_CLASSIFICATION::ENTROPY:
		if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
			return CalculateEntropyAndThresholdOrdinal(dataset, responses, score_and_threshold, 0);
		else
			return CalculateEntropyAndThreshold(dataset, responses, score_and_threshold, 0);
		break;
	case SPLITEVALUATION_TYPE_CLASSIFICATION::GINI:
		if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
			return CalculateEntropyAndThresholdOrdinal(dataset, responses, score_and_threshold, 1);
		else
			return CalculateEntropyAndThreshold(dataset, responses, score_and_threshold, 1);
		break;
	case SPLITEVALUATION_TYPE_CLASSIFICATION::LOSS_SPECIFIC:
		if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
			throw std::logic_error("SplitEvaluatorMLClass: ordinal splits not implemented for loss-specific splitting!");
		return CalculateSpecificLossAndThreshold(dataset, responses, score_and_threshold);
		break;
	default:
		throw std::runtime_error("SplitEvaluatorMLClass: splitfunction_classification_loss_type not defined");
		return false;
		break;
	}
}






// PROTECTED/HELPER METHODS
template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLClass<Sample, TAppContext>::CalculateEntropyAndThreshold(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold, int use_gini)
{
	// In: samples, sorted responses, out: optimality-measure + threshold

    // Initialize the counters
    double DGini, LGini, RGini, LTotal = 0.0, RTotal = 0.0, bestThreshold = 0.0, bestDGini = 1e16;
    vector<double> LCount(m_appcontext->num_classes, 0.0), RCount(m_appcontext->num_classes, 0.0);
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
    	int labelIdx = dataset[responses[r].second]->m_label.class_label;
    	double sample_w = dataset[responses[r].second]->m_label.class_weight;

    	RCount[labelIdx] += sample_w;
        RTotal += sample_w;
    }

    // Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
    int th_idx = 0;
    bool stop_search = false;
    for (int r = 0; r < responses.size(); r++)
    {
        // if the current sample is smaller than the current threshold put it to the left side
        if (responses[r].first <= random_thresholds[th_idx])
        {
            double cur_sample_weight = dataset[responses[r].second]->m_label.class_weight;

            RTotal -= cur_sample_weight;
            if (RTotal < 0.0)
            	RTotal = 0.0;
            LTotal += cur_sample_weight;
            int labelIdx = dataset[responses[r].second]->m_label.class_label;
            RCount[labelIdx] -= cur_sample_weight;
            if (RCount[labelIdx] < 0.0)
            	RCount[labelIdx] = 0.0;
            LCount[labelIdx] += cur_sample_weight;
        }
        else
        {
            // ok, now we found the first sample having higher response than the current threshold

            // now, we have to check the Gini index, this would be a valid split
            LGini = 0.0, RGini = 0.0;
            if (use_gini)
            {
                for (int c = 0; c < LCount.size(); c++)
                {
                    double pL = LCount[c]/LTotal, pR = RCount[c]/RTotal;
                    if (LCount[c] >= 1e-10) // FUCK YOU rounding errors
                        LGini += pL * (1.0 - pL);
                    if (RCount[c] >= 1e-10)
                        RGini += pR * (1.0 - pR);
                }
            }
            else
            {
                for (int c = 0; c < LCount.size(); c++)
                {
                    double pL = LCount[c]/LTotal, pR = RCount[c]/RTotal;
                    if (LCount[c] >= 1e-10) // FUCK YOU rounding errors
                        LGini -= pL * log(pL);
                    if (RCount[c] >= 1e-10)
                        RGini -= pR * log(pR);
                }
            }
            DGini = (LTotal*LGini + RTotal*RGini)/(LTotal + RTotal);

            if (DGini < bestDGini && LTotal > 0.0 && RTotal > 0.0)
            {
                bestDGini = DGini;
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

    score_and_threshold.first = bestDGini;
    score_and_threshold.second = bestThreshold;
    return found;
}


template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLClass<Sample, TAppContext>::CalculateEntropyAndThresholdOrdinal(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold, int use_gini)
{
	// In: samples, sorted responses, out: optimality-measure + threshold

	double bestDGini = 1e16, bestThreshold = 0.0;
	bool found = false;

	// iterate the ordinal "thresholds", i.e., the indices 0 ... K-1
	for (int t = 0; t < m_appcontext->ordinal_split_k; t++)
	{
		// now, find all samples with responses[:].first == t, and the other ones
		vector<double> LCount(m_appcontext->num_classes, 0.0), RCount(m_appcontext->num_classes, 0.0);
		double LTotal = 0.0, RTotal = 0.0;
		for (size_t s = 0; s < responses.size(); s++)
		{
			// get the infos of the current sample
			int labelIdx = dataset[responses[s].second]->m_label.class_label;
			double sample_w = dataset[responses[s].second]->m_label.class_weight;

			// check "threshold"
			if ((int)responses[s].first == t)
			{
				LCount[labelIdx] += sample_w;
				LTotal += sample_w;
			}
			else
			{
				RCount[labelIdx] += sample_w;
				RTotal += sample_w;
			}
		}

		// calculate purity measure (entropy or Gini)
		double LGini = 0.0, RGini = 0.0;
		if (use_gini)
		{
			for (int c = 0; c < LCount.size(); c++)
			{
				double pL = LCount[c]/LTotal, pR = RCount[c]/RTotal;
				if (LCount[c] >= 1e-10) // FUCK YOU rounding errors
					LGini += pL * (1.0 - pL);
				if (RCount[c] >= 1e-10)
					RGini += pR * (1.0 - pR);
			}
		}
		else
		{
			for (int c = 0; c < LCount.size(); c++)
			{
				double pL = LCount[c]/LTotal, pR = RCount[c]/RTotal;
				if (LCount[c] >= 1e-10) // FUCK YOU rounding errors
					LGini -= pL * log(pL);
				if (RCount[c] >= 1e-10)
					RGini -= pR * log(pR);
			}
		}
		double DGini = (LTotal*LGini + RTotal*RGini)/(LTotal + RTotal);

		if (DGini < bestDGini && LTotal > 0.0 && RTotal > 0.0)
		{
			bestDGini = DGini;
			bestThreshold = (double)t;
			found = true;
		}

	}

    score_and_threshold.first = bestDGini;
    score_and_threshold.second = bestThreshold;
    return found;
}


template<typename Sample, typename TAppContext>
bool SplitEvaluatorMLClass<Sample, TAppContext>::CalculateSpecificLossAndThreshold(DataSet<Sample, LabelMLClass>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	// In: samples, sorted responses, out:loss-value+threshold

    // 1) Calculate random thresholds and sort them
    double min_response = responses[0].first;
    double max_response = responses[responses.size()-1].first;
    double d = (max_response - min_response);
    vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
    for (int i = 0; i < random_thresholds.size(); i++)
        random_thresholds[i] = (randDouble() * d) + min_response;
    sort(random_thresholds.begin(), random_thresholds.end());


    // Declare and init some variables
    vector<double> RClassWeights(m_appcontext->num_classes, 0.0);
    vector<double> LClassWeights(m_appcontext->num_classes, 0.0);
    vector<int> RSamples;
    vector<int> LSamples;
    double RTotalWeight = 0.0;
    double LTotalWeight = 0.0;
    double margin = 0.0;
    double RLoss = 0.0, LLoss = 0.0;
    double BestLoss = 1e16, CombinedLoss = 0.0, TotalWeight = 0.0, BestThreshold = 0.0;
    bool found = false;


    // First, put everything in the right node
    RSamples.resize(responses.size());
    for (int r = 0; r < responses.size(); r++)
    {
        int labelIdx = dataset[responses[r].second]->m_label.class_label;
        double sample_w = dataset[responses[r].second]->m_label.class_weight;

        RClassWeights[labelIdx] += sample_w;
        RTotalWeight += sample_w;
        RSamples[r] = responses[r].second;
    }

    // Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
    int th_idx = 0;
    bool stop_search = false;
    for (int r = 0; r < responses.size(); r++)
    {
        // if the current sample is smaller than the current threshold put it to the left side
        if (responses[r].first <= random_thresholds[th_idx])
        {
            int labelIdx = dataset[responses[r].second]->m_label.class_label;
            double cur_sample_weight = dataset[responses[r].second]->m_label.class_weight;

            RClassWeights[labelIdx] -= cur_sample_weight;
            if (RClassWeights[labelIdx] < 0.0)
                RClassWeights[labelIdx] = 0.0;
            LClassWeights[labelIdx] += cur_sample_weight;

            RTotalWeight -= cur_sample_weight;
            if (RTotalWeight < 0.0)
                RTotalWeight = 0.0;
            LTotalWeight += cur_sample_weight;

            LSamples.push_back(RSamples[0]);
            RSamples.erase(RSamples.begin());
        }
        else
        {
            // ok, now we found the first sample having higher response than the current threshold

            // Reset the losses
            RLoss = 0.0, LLoss = 0.0;

            // calculate loss for left and right child nodes
            // RIGHT
			vector<double> pR(RClassWeights.size());
			for (int ci = 0; ci < RClassWeights.size(); ci++)
				pR[ci] = RClassWeights[ci] / RTotalWeight;
			for (int ci = 0; ci < RClassWeights.size(); ci++)
				RLoss += RClassWeights[ci] * ComputeLoss(pR, ci, m_appcontext->global_loss_classification);

            // LEFT
            vector<double> pL(LClassWeights.size());
			for (int ci = 0; ci < LClassWeights.size(); ci++)
				pL[ci] = LClassWeights[ci] / LTotalWeight;
			for (int ci = 0; ci < LClassWeights.size(); ci++)
				LLoss += LClassWeights[ci] * ComputeLoss(pL, ci, m_appcontext->global_loss_classification);

            // Total loss
            CombinedLoss = LLoss + RLoss;

            // best-search ...
            if (CombinedLoss < BestLoss && LTotalWeight > 0.0 && RTotalWeight > 0.0)
            {
                BestLoss = CombinedLoss;
                BestThreshold = random_thresholds[th_idx];
                found = true;
            }

            // next, we have to find the next random threshold that is larger than the current response
            // -> there might be several threshold within the gap between the last response and this one.
            while (responses[r].first > random_thresholds[th_idx])
            {
                if (th_idx < (random_thresholds.size()-1))
                {
                    th_idx++;
                    r--;
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

    score_and_threshold.first = BestLoss;
    score_and_threshold.second = BestThreshold;
    return found;
}


template<typename Sample, typename TAppContext>
double SplitEvaluatorMLClass<Sample, TAppContext>::ComputeLoss(vector<double> p, int c, ADF_LOSS_CLASSIFICATION::Enum wut)
{
    // This is for the GRAD_* losses
    double margin, margin_scale = 2.0;
    //if (m_appcontext->splitfunction_scalemargin)
	margin = margin_scale * (p[c]-0.5);

    switch (wut)
    {
    case ADF_LOSS_CLASSIFICATION::GRAD_LOGIT:
    {
        return log(1.0 + exp(-margin));
        break;
    }
    case ADF_LOSS_CLASSIFICATION::GRAD_HINGE:
    {
        return max(0.0, 1.0 - margin);
        break;
    }
    case ADF_LOSS_CLASSIFICATION::GRAD_SAVAGE:
    {
        return 1.0 / pow((1.0 + exp(2.0*margin)), 2.0);
        break;
    }
    case ADF_LOSS_CLASSIFICATION::GRAD_EXP:
    {
        return exp(-margin);
        break;
    }
    case ADF_LOSS_CLASSIFICATION::GRAD_TANGENT:
    {
        return pow(2.0 * atan(margin) - 1.0, 2.0);
        break;
    }
    case ADF_LOSS_CLASSIFICATION::ZERO_ONE:
        // For the ZERO_ONE loss, the scaling has no influence!
        int max_c = -1;
        double max_p = -1.0;
        for (int i = 0; i < p.size(); i++)
        {
            if (p[i] > max_p)
            {
                max_p = p[i];
                max_c = i;
            }
        }
        if (max_c == c)
            return 0.0;
        else
            return 1.0;
        break;
    }
}


#endif /* SPLITEVALUATORMLCLASS_CPP_ */
