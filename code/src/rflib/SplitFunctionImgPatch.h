/*
 * SplitFunctionImgPatch.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITFUNCTIONIMGPATCH_H_
#define SPLITFUNCTIONIMGPATCH_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include "opencv2/opencv.hpp"

#include "AppContext.h"
#include "SampleImgPatch.h"

#include "icgrf.h"

using namespace std;
using namespace Eigen;



template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
class SplitFunctionImgPatch
{
public:
	SplitFunctionImgPatch(AppContext* appcontextin);
	virtual ~SplitFunctionImgPatch();
	void SetRandomValues();
	void SetThreshold(double inth);
	void SetSplit(SplitFunctionImgPatch* spfin);
	int Split(SampleImgPatch& sample);
	double CalculateResponse(SampleImgPatch& sample);

    // analysis method
    void Print();

	void Save(std::ofstream& out);
	void Load(std::ifstream& in);

	int GetCh1() const
	{
		return ch1;
	}

	int GetCh2() const
	{
		return ch2;
	}

	cv::Point GetPx1() const
	{
		return px1;
	}

	cv::Point GetPx2() const
	{
		return px2;
	}

	cv::Rect GetRe1() const
	{
		return re1;
	}

	cv::Rect GetRe2() const
	{
		return re2;
	}

protected:
	double GetResponse(SampleImgPatch& sample);
	AppContext* m_appcontext;
	double m_th;
	int ch1;
	cv::Point px1;
	int ch2;
	cv::Point px2;
	cv::Rect re1;
	cv::Rect re2;
	std::vector<cv::Point> pxs;

    // the type of this split function
    int m_splitfunction_type;
};

#include "SplitFunctionImgPatch.cpp"


#endif /* SPLITFUNCTIONIMGPATCH_H_ */
