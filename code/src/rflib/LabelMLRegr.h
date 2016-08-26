/*
 * LabelMLRegr.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LABELMLREGR_H_
#define LABELMLREGR_H_

#include <stdexcept>
#include <vector>
#include "eigen3/Eigen/Core"


struct LabelMLRegr
{
	LabelMLRegr();

	double regr_weight_gt;
	double regr_weight; // working weight
	Eigen::VectorXd regr_target_gt;
	Eigen::VectorXd regr_target; // working target

	void Save(std::ofstream& out);
	void Load(std::ifstream& in);
};

#endif /* LABELMLREGR_H_ */
