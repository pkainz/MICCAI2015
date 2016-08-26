/*
 * CellDetectorRegr.h
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#ifndef CELLDETECTORREGR_H
#define CELLDETECTORREGR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

#include "AppContextCellDetRegr.h"
#include "SampleImgPatch.h"
#include "LabelMLRegr.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorMLRegr.h"
#include "LeafNodeStatisticsMLRegr.h"
#include "DataLoaderCellDetRegr.h"

#include "icgrf.h"

// typedefs for easier use later
typedef SplitFunctionImgPatch<uchar, float, AppContextCellDetRegr> TSplitFunctionImgPatch;
typedef RandomForest<SampleImgPatch, LabelMLRegr, TSplitFunctionImgPatch, SplitEvaluatorMLRegr<SampleImgPatch, AppContextCellDetRegr>, LeafNodeStatisticsMLRegr<SampleImgPatch, AppContextCellDetRegr>, AppContextCellDetRegr> TRegressionForest;
typedef Node<SampleImgPatch, LabelMLRegr, TSplitFunctionImgPatch, LeafNodeStatisticsMLRegr<SampleImgPatch, AppContextCellDetRegr>, AppContextCellDetRegr> TNode;

class CellDetectorRegr {

public:
    // Constructor
    CellDetectorRegr(TRegressionForest* rfin, AppContextCellDetRegr* apphpin);

    // detect cells from test images (uses test path from appcontext)
    void PredictTestImages();

    // predict a single image
    cv::Mat PredictSingleImage(const boost::filesystem::path file);

protected:
    // predict the distance transform image using sliding window
    // and store the prediction image to a specified folder
    void PredictImage(const cv::Mat& src_img, cv::Mat& pred_dt_img);

private:
    TRegressionForest* m_rf; // the forest
    AppContextCellDetRegr* m_apphp; // the app context
    int m_pwidth; // patch width
    int m_pheight; // patch height
};


#endif // CELLDETECTORREGR_H
