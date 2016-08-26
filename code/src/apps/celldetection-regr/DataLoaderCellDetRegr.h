/*
 * DataLoaderCellDetRegr.cpp
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#ifndef DATALOADERCELLDETREGR_H
#define DATALOADERCELLDETREGR_H

#include <vector>
#include <eigen3/Eigen/Core>
#include "opencv2/opencv.hpp"

#include <boost/regex.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "icgrf.h"

#include "AppContextCellDetRegr.h"
#include "SampleImgPatch.h"
#include "LabelMLRegr.h"
#include "FeatureGeneratorRGBImage.h"

#include "CellDetUtils.h"

using namespace std;
using namespace Eigen;


class DataLoaderCellDetRegr
{
public:

    // Constructors & Destructors
    explicit DataLoaderCellDetRegr(AppContextCellDetRegr* hp);
    ~DataLoaderCellDetRegr();

    // Loading dataset (for classification and regression tasks)
    DataSet<SampleImgPatch, LabelMLRegr> LoadTrainData(std::vector<boost::filesystem::path>& trainImgFilenames);
    void GetTrainDataProperties(int& num_samples, int& num_classes, int& num_target_variables, int& num_feature_channels);

    // static image & feature methods
    static cv::Mat ReadImageData(std::string imgpath, bool loadasgray);
    static void ExtractFeatureChannelsObjectDetection(const cv::Mat& img, std::vector<cv::Mat>& vImg, AppContextCellDetRegr* appcontext);

    // stored image data
    static std::vector<std::vector<cv::Mat> > image_feature_data;

private:

    Eigen::MatrixXd GeneratePatchLocations(const cv::Mat& src_img, const cv::Mat& dt_img);
    void SavePatchPositions(std::string savepath, std::vector<Eigen::MatrixXd> patch_positions);
    std::vector<Eigen::MatrixXd> LoadPatchPositions(std::string loadpath);

    void ExtractPatches(DataSet<SampleImgPatch, LabelMLRegr>& dataset, const cv::Mat& img, const std::vector<cv::Mat>& img_features, const Eigen::MatrixXd& patch_locations, int img_index, const cv::Mat& dt_img);

    // method for loading ground truth positions
    Eigen::MatrixXi LoadGTPositions(std::string loadpath);

    // write the detection results to a txt file
    void SaveDetectionPositions(std::string savepath, Eigen::MatrixXi detected_locations);

    // parameters
    AppContextCellDetRegr* m_hp;

    // a copy of the data set is also stored here. The dataset only contains pointers to the real data,
    // which is only stored once!
    DataSet<SampleImgPatch, LabelMLRegr> m_trainset;

    // data properties
    int m_num_samples;
    int m_num_classes;
    int m_num_target_variables;
    int m_num_feature_channels;
};

#endif // DATALOADERCELLDETREGR_H
