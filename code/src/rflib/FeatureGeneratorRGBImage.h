/*
 * FeatureGeneratorRGBImage.h
 *
 *  Created on: Sep 7, 2011
 *      Author: Matthias Dantone
 *
 */

#ifndef FEATUREGENERATORRGBIMAGE_H_
#define FEATUREGENERATORRGBIMAGE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdexcept>
//#include <math.h>
#include <cmath>

// TODO: this should possibly go to the configuration file?!?!
#define FC_GRAY     0
#define FC_GABOR    1
#define FC_SOBEL    2 // first derivatives
#define FC_MIN_MAX  3
#define FC_CANNY    4
#define FC_NORM     5
#define FC_LAB		6
#define FC_GRAD2	7 // second derivatives
#define FC_HOG		8
#define FC_LUV      9
#define FC_ORIENTED_GRAD_CHANFTRS 10
#define FC_GRADMAGN 11
#define FC_RGB		12
#define FC_RELLOC   13 // relative location



class FeatureGeneratorRGBImage
{
public:

  FeatureGeneratorRGBImage();

  void ExtractChannel(int type, bool useIntegral, const cv::Mat& src, std::vector<cv::Mat>& channels);

private:

  // stuff for the Gabor features
  void Gabor_transform(const cv::Mat& src, cv::Mat* dst, bool useIntegral, int index, int old_size) const;
  void Init_gabor_kernels();
  void CreateKernel(int iMu, int iNu, double sigma, double dF);
  std::vector<cv::Mat> reals;
  std::vector<cv::Mat> imags;

  // stuff for the HoG features
  void CalculateHoGGall(cv::Mat sob_x, cv::Mat sob_y, std::vector<cv::Mat>& hoglayers, int nbins = 9, int gauss_w = 5);

};

#endif /* FEATUREGENERATORRGBIMAGE_H_ */
