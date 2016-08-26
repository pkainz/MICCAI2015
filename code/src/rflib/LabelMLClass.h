/*
 * LabelMLClass.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LABELMLCLASS_H_
#define LABELMLCLASS_H_


#include <fstream>
#include <stdexcept>
#include <vector>


class LabelMLClass
{
public:
	LabelMLClass();

	double class_weight_gt;
	double class_weight; // working weight!
	int gt_class_label;
	int class_label; // working class label!

	// I/O stuff
	void Save(std::ofstream& out);
	void Load(std::ifstream& in);
};


#endif /* LABELMLCLASS_H_ */
