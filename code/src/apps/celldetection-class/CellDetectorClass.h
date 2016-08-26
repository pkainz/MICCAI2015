/*
 * CellDetectorClass.cpp
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#ifndef CELLDETECTORRCLASS_H
#define CELLDETECTORRCLASS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

#include "AppContextCellDetClass.h"
#include "SampleImgPatch.h"
#include "LabelMLClass.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorMLClass.h"
#include "LeafNodeStatisticsMLClass.h"
#include "DataLoaderCellDetClass.h"

#include "icgrf.h"

// typedefs for easier use later
typedef SplitFunctionImgPatch<uchar, float, AppContextCellDetClass> TSplitFunctionImgPatch;
typedef RandomForest<SampleImgPatch, LabelMLClass, TSplitFunctionImgPatch, SplitEvaluatorMLClass<SampleImgPatch, AppContextCellDetClass>, LeafNodeStatisticsMLClass<SampleImgPatch, AppContextCellDetClass>, AppContextCellDetClass> TClassificationForest;
typedef Node<SampleImgPatch, LabelMLClass, TSplitFunctionImgPatch, LeafNodeStatisticsMLClass<SampleImgPatch, AppContextCellDetClass>, AppContextCellDetClass> TNode;

class CellDetectorClass {

public:
    // Constructor
    CellDetectorClass(TClassificationForest* rfin, AppContextCellDetClass* apphpin);

    // detect cells from test images (uses test path from appcontext)
    void PredictTestImages();

    // predict a single image
    cv::Mat PredictSingleImage(const boost::filesystem::path file);

protected:
    // predict the distance transform image using sliding window
    // and store the prediction image to a specified folder
    void PredictImage(const cv::Mat& src_img, cv::Mat& pred_img);

private:
    TClassificationForest* m_rf; // the forest
    AppContextCellDetClass* m_apphp; // the app context
    int m_pwidth; // patch width
    int m_pheight; // patch height
};


#endif // CELLDETECTORCLASS_H
