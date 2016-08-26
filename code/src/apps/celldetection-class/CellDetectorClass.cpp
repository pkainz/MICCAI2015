/*
 * CellDetectorClass.cpp
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#include "CellDetectorClass.h"

CellDetectorClass::CellDetectorClass(TClassificationForest* rfin, AppContextCellDetClass* apphp) : m_apphp(apphp), m_rf(rfin) {
    this->m_pwidth = m_apphp->patch_size[1];
    this->m_pheight = m_apphp->patch_size[0];
}

/**
 * Predict the distance transform image.
 *
 * @param src_img the source image
 * @param pred_conf_img the confidence map of this image
 */
void CellDetectorClass::PredictImage(const cv::Mat& src_img, cv::Mat& pred_img){

    if (!m_apphp->quiet)
        cout << "Predicting... " << flush;

    // 01) extract feature planes
    std::vector<cv::Mat> img_features;
    DataLoaderCellDetClass::ExtractFeatureChannelsObjectDetection(src_img, img_features, m_apphp);

    // 02) initialize the prediction image (forest predicts floats!)
    pred_img = cv::Mat::zeros(src_img.rows, src_img.cols, CV_32F);

    // 03) use the m_rf and predict each pixel position of the image
    // define the patch offsets
    int xoffset = (int)(m_pwidth/2);
    int yoffset = (int)(m_pheight/2);

    // Define the search region
    int startX = xoffset;
    int endX   = src_img.cols - xoffset;
    int startY = yoffset;
    int endY   = src_img.rows - yoffset;

    // set the pixel stride
    int pixel_step = 1;

    // iterate the image
    SampleImgPatch imgpatch;
    imgpatch.features = img_features;

    // ALWAYS! compute the normalization feature mask
    cv::Mat temp_mask;
    temp_mask.create(cv::Size(src_img.cols, src_img.rows), CV_8U);
    temp_mask.setTo(cv::Scalar::all(1.0));
    cv::integral(temp_mask, imgpatch.normalization_feature_mask, CV_32F);

    // we only have to fill the Sample in the labelled sample for testing ... so we use a dummy label
    LabelMLClass dummy_label = LabelMLClass();
    LabelledSample<SampleImgPatch, LabelMLClass>* labelled_sample =
            new LabelledSample<SampleImgPatch, LabelMLClass>(imgpatch, dummy_label, 1.0, 0);

    // move sliding window over the image
    // y,x is the center pixel position to be predicted
    float confidence_value_foreground;
    int predicted_class;
    for (int y = startY; y < endY; y += pixel_step)
    {
        for (int x = startX; x < endX; x += pixel_step)
        {
            // set the patch location in the image
            labelled_sample->m_sample.x = x-xoffset;
            labelled_sample->m_sample.y = y-yoffset;

            LeafNodeStatisticsMLClass<SampleImgPatch, AppContextCellDetClass> stats(m_apphp);
            stats = m_rf->TestAndAverage(labelled_sample);

//            for (size_t c = 0; c < stats.m_class_histogram.size(); c++)
//                cout << stats.m_class_histogram[c] << " ";

//            cout << endl;

            // get the maximum value from the histogram (this indicates the class)
            std::vector<double>::iterator result;
            result = std::max_element(stats.m_class_histogram.begin(), stats.m_class_histogram.end());
            predicted_class = std::distance(stats.m_class_histogram.begin(), result);

            // nevertheless just use the probability for foreground in the prediction image
            confidence_value_foreground = stats.m_class_histogram[1];
            pred_img.at<float>(y, x) = confidence_value_foreground;

            //std::cout << confidence_value_foreground << std::endl;
        }
        //cout << "max class response: " << predicted_class << ", fg-probability: ";
        //cout << confidence_value_foreground << endl;
    }

    if (!m_apphp->quiet)
        cout << " done." << endl;

    // delete the labelled sample, this is important otherwise, all the image features won't be free'd!
    delete(labelled_sample);

    {
        cout << "Clearing feature map of prediction image" << endl;
        std::vector<cv::Mat> tmp;
        img_features.clear();
        tmp.swap(img_features);

        std::vector<cv::Mat> tmp2;
        imgpatch.features.clear();
        tmp2.swap(imgpatch.features);
    }
}

// loads and predicts a single image
cv::Mat CellDetectorClass::PredictSingleImage(boost::filesystem::path file){
    string filename = file.filename().string();

    // Load image
    cv::Mat src_img_raw = DataLoaderCellDetClass::ReadImageData(file.c_str(), false);
    // scaling
    cv::Mat src_img;
    cv::Size new_size = cv::Size((int)((double)src_img_raw.cols*this->m_apphp->general_scaling_factor), (int)((double)src_img_raw.rows*this->m_apphp->general_scaling_factor));
    resize(src_img_raw, src_img, new_size, 0, 0, cv::INTER_LINEAR);

    // extend border if required
    ExtendBorder(m_apphp, src_img, src_img, true);

    // The predicted image
    cv::Mat pred_img;

    // predict the image
    this->PredictImage(src_img, pred_img);

    //        cv::namedWindow("prediction", CV_WINDOW_AUTOSIZE );
    //        cv::imshow("prediction", _8bit);
    //        cv::waitKey(0);

    // store the predicted image to the hd
    if (m_apphp->store_predictionimages == 1)
    {
        if (!m_apphp->quiet){
            cout << "Storing prediction image to " << m_apphp->path_predictionimages << filename << endl;
        }
        // convert output image
        cv::Mat tmp;
        cv::normalize(pred_img, pred_img, 0, 255, cv::NORM_MINMAX, CV_8U);
        pred_img.convertTo(tmp, CV_8U);
        cv::imwrite(m_apphp->path_predictionimages + filename, tmp);
    }

    return pred_img;
}

/**
 * Loads the images from a directory and produces the prediction results.
 * Post-processing is done in matlab.
 */
void CellDetectorClass::PredictTestImages(){

    // 01) read all images from the test path
    vector<boost::filesystem::path> testImgFilenames;
    testImgFilenames.clear();
    ls(m_apphp->path_testdata, ".*png", testImgFilenames);

    // 1) read the source data
    if (!m_apphp->quiet)
        cout << "Using " << testImgFilenames.size() << " images for testing ..." << endl;

    // Run detector for each image
#pragma omp parallel for
    for (size_t i = 0; i < testImgFilenames.size(); i++)
    {
        this->PredictSingleImage(testImgFilenames[i]);

        // Status message
        if (!m_apphp->quiet)
            cout << "Done predicting image " << i + 1 << " / " << testImgFilenames.size() << endl;

    }
    if (!m_apphp->quiet)
        cout << "Done predicting all images." << endl;
}
