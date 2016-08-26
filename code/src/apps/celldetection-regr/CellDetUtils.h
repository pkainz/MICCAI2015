/*
 * CellDetUtils.h
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#ifndef CELLDETUTILS_H
#define CELLDETUTILS_H


#include <stdio.h>
#include <iostream>
#include <limits>

#include <boost/regex.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <opencv2/opencv.hpp>

#include "AppContextCellDetRegr.h"

using namespace std;

/**
 * List all files in a specific directory.
 *
 * @param dir the directory
 * @param regex file name regex-pattern
 * @param files the vector of files
 */
inline void ls(boost::filesystem::path dir, std::string regex, std::vector<boost::filesystem::path>& files) {
    files.clear();
    boost::regex expression(regex);
    for ( boost::filesystem::recursive_directory_iterator end, current(dir); current != end; ++current ) {
        boost::filesystem::path cp = current->path();

        boost::cmatch what;
        std::string cpStr = cp.filename().string();
        if(boost::regex_match(cpStr.c_str(), what, expression)) {
            files.push_back(cp);
        }
    }

    // sort the files
    std::sort(files.begin(), files.end());
}

/**
 * Labels all blobs in a binary image
 *
 * @param binary
 * @param blobs
 */
inline void LabelBlobs(const cv::Mat& binary, std::vector<std::vector<cv::Point> >& blobs)
{
    blobs.clear();

    // Using labels from 2+ for each blob
    cv::Mat label_image;
    binary.convertTo(label_image, CV_32FC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if((int)label_image.at<float>(y,x) != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);

            std::vector<cv::Point> blob;
            blob.clear();

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if((int)label_image.at<float>(i,j) != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

inline void ListAllTrainImgFilenames(AppContextCellDetRegr* apphp, std::vector<boost::filesystem::path>& trainImgFilenames){
    trainImgFilenames.clear();
    ls(apphp->path_traindata, ".*png", trainImgFilenames);
}

inline void ListAllTestImgFilenames(AppContextCellDetRegr* apphp, std::vector<boost::filesystem::path>& testImgFilenames){
    testImgFilenames.clear();
    ls(apphp->path_testdata, ".*png", testImgFilenames);
}

inline void ExtendBorder(AppContextCellDetRegr* apphp, cv::Mat& src_img, cv::Mat& dst_img, bool rgb){
    if (apphp->extend_border){
        int padding = (int) apphp->patch_size[0]/2;

        switch(apphp->border_type){
        // cv::BORDER_REPLICATE = 0,
        // cv::BORDER_CONSTANT=1,
        // cv::BORDER_REFLECT=2,
        // cv::BORDER_WRAP=3,
        // cv::BORDER_REFLECT_101 = 4
        case 0:
            cv::copyMakeBorder(src_img, dst_img, padding, padding, padding, padding, cv::BORDER_REPLICATE);
            break;
        case 1:
            cv::copyMakeBorder(src_img, dst_img, padding, padding, padding, padding, cv::BORDER_CONSTANT, rgb?cv::Scalar(0,0,0):cv::Scalar(0));
            break;
        case 2:
            cv::copyMakeBorder(src_img, dst_img, padding, padding, padding, padding, cv::BORDER_REFLECT);
            break;
        case 3:
            cv::copyMakeBorder(src_img, dst_img, padding, padding, padding, padding, cv::BORDER_WRAP);
            break;
        case 4:
            cv::copyMakeBorder(src_img, dst_img, padding, padding, padding, padding, cv::BORDER_REFLECT_101);
            break;
        default:
            cout << "Unrecognized border type: " << apphp->border_type << endl;
            exit(-1);
        }
    }
}

#endif // CELLDETUTILS_H
