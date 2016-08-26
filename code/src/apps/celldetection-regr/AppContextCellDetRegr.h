/*
 * AppContextCellDetRegr.h
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#ifndef APPCONTEXTCELLDETREGR_H
#define APPCONTEXTCELLDETREGR_H

#include <ostream>
#include <string>
#include <vector>
#include <libconfig.h++>
#include "icgrf.h"

#include "AppContext.h"
#include "FeatureGeneratorRGBImage.h"

using std::cout;
using std::endl;


class AppContextCellDetRegr : public AppContext
{
public:
    AppContextCellDetRegr() {}
    virtual ~AppContextCellDetRegr() {}

    // Method for printing the Hyperparameters
    friend std::ostream& operator<< (std::ostream &o, const AppContextCellDetRegr& apphp)
    {
        o << "===========================================================" << std::endl;
        o << "Application Context Parameters (CELL DETECTION REGRESSION)" << std::endl;
        o << "===========================================================" << std::endl;

        o << "quiet                    = " << apphp.quiet                   << std::endl;
        o << "debug_on                 = " << apphp.debug_on			   << std::endl << std::endl;

        string mode = "";

        switch(apphp.mode){
        case 0:
            mode = "train";
            break;
        case 1:
            mode = "predict test";
            break;
        case 2:
            mode = "train and predict test";
            break;
        case 3:
            mode = "analyze leaf nodes";
            break;
        case 4:
            mode = "analyze split nodes";
            break;
        case 5:
            mode = "Leave-one-Out Cross Validation";
            break;
        default:
            mode = "UNDEFINED!";
            break;
        }
        o << "mode                     = " << mode                            << std::endl << std::endl;

        o << "num_trees                = " << apphp.num_trees                 << std::endl;
        o << "max_tree_depth           = " << apphp.max_tree_depth            << std::endl;
        o << "min_split_samples        = " << apphp.min_split_samples         << std::endl;
        o << "num_node_tests           = " << apphp.num_node_tests            << std::endl;
        o << "num_node_thresholds      = " << apphp.num_node_thresholds       << std::endl << std::endl;

        o << "image_feature_list (size)= " << apphp.image_feature_list.size() << std::endl;
        int currChannel = -1;


        bool use_min_max = (apphp.use_min_max_filters==1);
        o << "use_min_max_filters      = " << use_min_max  << std::endl;

        for (size_t i = 0; i < apphp.image_feature_list.size(); i++){

            // resolve the names of the features (number of channels may not be accurate!!)
            switch(apphp.image_feature_list[i]){

            case FC_GRAY:
                o << "[0] = Gray                                ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?0:1;
                o << " - " << currChannel;
                break;

            case FC_GABOR:
                o << "[1] = Gabor                               ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?24:49;
                o << " - " << currChannel;
                break;

            case FC_SOBEL:
                o << "[2] = Sobel (Gradients)                   ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?1:3;
                o << " - " << currChannel;
                break;

            case FC_MIN_MAX:
                o << "[3] = MinMax                              ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?1:3;
                o << " - " << currChannel;
                break;

            case FC_CANNY:
                o << "[4] = Canny Edges                         ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?0:1;
                o << " - " << currChannel;
                break;

            case FC_NORM:
                o << "[5] = Normalize (histeq)                  ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?0:1;
                o << " - " << currChannel;
                break;

            case FC_LAB:
                o << "[6] = L*a*b Channels                      ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?2:5;
                o << " - " << currChannel;
                break;

            case FC_GRAD2:
                o << "[7] = 2nd order gradients (GRAD2)         ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?1:3;
                o << " - " << currChannel;
                break;
            case FC_HOG:
                o << "[8] = HoG                                 ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?8:17;
                o << " - " << currChannel;
                break;

            case FC_LUV:
                o << "[9] = L*u*v                               ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?2:5;
                o << " - " << currChannel;
                break;

            case FC_ORIENTED_GRAD_CHANFTRS:
                o << "[10] = Oriented Gradient Channel Filters  ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?5:11;
                o << " - " << currChannel;
                break;

            case FC_GRADMAGN:
                o << "[11] = Gradient Magnitude                 ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?0:1;
                o << " - " << currChannel;
                break;

            case FC_RGB:
                o << "[12] = RGB                                ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?2:5;
                o << " - " << currChannel;
                break;

            case FC_RELLOC:
                o << "[13] = Relative Location                  ";
                currChannel += 1;
                o << ", channels " << currChannel;
                currChannel += !use_min_max?1:3;
                o << " - " << currChannel;
                break;
            default:

                o << "[??] UNDEFINED                            ";
                break;
            }

            o << endl;
        }
        o << "num_feature_planes       = " << (currChannel+1)     << std::endl;

        o << "split_eval_type_regr     = ";
        switch(apphp.splitevaluation_type_regression){
        case SPLITEVALUATION_TYPE_REGRESSION::REDUCTION_IN_VARIANCE:
            o << "REDUCTION_IN_VARIANCE";
            break;
        case SPLITEVALUATION_TYPE_REGRESSION::DIFF_ENTROPY_GAUSS:
            o << "DIFF_ENTROPY_GAUSS";
            break;

        case SPLITEVALUATION_TYPE_REGRESSION::DIFF_ENTROPY_GAUSS_BLOCK_POSE:
            o << "DIFF_ENTROPY_GAUSS_BLOCK_POSE";
            break;

        default:
            o << "UNDEFINED!";
            break;
        }
        o << std::endl;

        o << "leaf_node_regr_type      = ";
        switch(apphp.leafnode_regression_type){
        case LEAFNODE_REGRESSION_TYPE::MEAN:
            o << "MEAN";
            break;
        case LEAFNODE_REGRESSION_TYPE::HILLCLIMB:
            o << "HILLCLIMB";
            break;
        case LEAFNODE_REGRESSION_TYPE::MEANSHIFT:
            o << "MEANSHIFT";
            break;
        default:
            o << "UNDEFINED!";
            break;
        }
        o << std::endl;

        o << "patch_size               = " << apphp.patch_size[0] << "x" << apphp.patch_size[1] << std::endl;
        o << "extend_border            = " << apphp.extend_border << std::endl;
        o << "border_type              = ";

        string bt = "";
        switch(apphp.border_type){
        // cv::BORDER_REPLICATE = 0,
        // cv::BORDER_CONSTANT=1,
        // cv::BORDER_REFLECT=2,
        // cv::BORDER_WRAP=3,
        // cv::BORDER_REFLECT_101 = 4
        case 0:
            bt ="cv::BORDER_REPLICATE";
            break;
        case 1:
            bt ="cv::BORDER_CONSTANT";
            break;

        case 2:
            bt ="cv::BORDER_REFLECT";
            break;

        case 3:
            bt ="cv::BORDER_WRAP";
            break;

        case 4:
            bt ="cv::BORDER_REFLECT_101";
            break;

        default:
            bt = "UNDEFINED!";
            break;
        }

        o  << bt << std::endl << std::endl;

        o << "DATA AND SAMPLING SETTINGS" << std::endl;
        o << "store_dataset            = " << (apphp.store_dataset?"true":"false") << std::endl;
        o << "load_dataset             = " << (apphp.load_dataset?"true":"false") << std::endl;
        o << "path_fixed_dataset       = " << apphp.path_fixedDataset << std::endl;
        o << "ratio_add_bg_patches     = " << apphp.ratio_additional_bg_patches << std::endl;
        o << "threshold_fg_bg          = " << apphp.threshold_fg_bg << std::endl;
        o << "path_traindata           = " << apphp.path_traindata << std::endl;
        o << "path_trainlabels         = " << apphp.path_trainlabels << std::endl;
        o << "path_testdata            = " << apphp.path_testdata << std::endl;
        o << "path_testlabels          = " << apphp.path_testlabels << std::endl;
        o << "use_unique_loocv_path    = " << apphp.use_unique_loocv_path << std::endl;
        o << "path_loocv               = " << apphp.path_loocv << std::endl;
        o << "loocv_path_pred_prefix   = " << apphp.loocv_path_predictions_prefix << std::endl;
        o << "loocv_path_det_prefix    = " << apphp.loocv_path_detections_prefix << std::endl;
        o << "loocv_path_trees_prefix  = " << apphp.loocv_path_trees_prefix << std::endl;
    }

protected:
    // implements the abstract base method!
    inline void ValidateHyperparameters()
    {
        if (!AppContext::ValidateCompleteGeneralPart())
        {
            cout << "General settings missing!" << endl;
            exit(-1);
        }

        if (!AppContext::ValidateStandardForestSettings())
        {
            cout << "Standard Forest settings missing" << endl;
            exit(-1);
        }

        if (this->leafnode_regression_type == LEAFNODE_REGRESSION_TYPE::NOTSET)
        {
            cout << "specify a leafnode-regression type!" << endl;
            exit(-1);
        }

        if (!AppContext::ValidateImageSplitFunction())
        {
            cout << "you have to specify a split function suitable for images!" << endl;
            exit(-1);
        }

        if (this->method != RF_METHOD::STDRF)
        {
            cout << "You have to use: STDRF (0)" << endl;
            exit(-1);
        }

        if (this->mode > 5) {
            cout << "You must specify a mode smaller than 5!" << endl;
            exit(-1);
        }

        if (!this->ValidateDetectionSpecificStuff())
        {
            cout << "Some detection specific parameters are missing in the config file!" << endl;
            exit(-1);
        }

        if (!this->ValidateDetectionDataStuff())
        {
            cout << "Some detection specific data parameters are missing!" << endl;
            exit(-1);
        }

        if (!this->ValidateCellDetectionRegression()){
            cout << "Some cell detection REGRESSION parameters are missing!" << endl;
            exit(-1);
        }


        if (this->store_dataset && this->load_dataset)
        {
            cout << "WARNING: Storing & Loading fixed data set is active: turning OFF store_dataset" << endl;
            this->store_dataset = 0;
        }
    }


    inline bool ValidateDetectionSpecificStuff()
    {
        if (this->patch_size.size() == 0)
        {
            std::cout << "patch size" << std::endl;
            return false;
        }

        if (this->extend_border < 0){
            std::cout << "extend_border is not set" << std::endl;
            return false;
        } else {
            if (this->extend_border == 1){
                if (!(this->border_type > 0 && this->border_type < 5)){
                    std::cout << "unrecognized border_type" << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    inline bool ValidateDetectionDataStuff()
    {
        if (this->mode == 5) // loocv
        {
            if (this->path_loocv.empty()){
                std::cout << "path_loocv is not set or empty." << std::endl;
                return false;
            }

            if (this->loocv_path_predictions_prefix.empty()){
                std::cout << "loocv_path_predictions_prefix is not set or empty, using default 'predictions'." << std::endl;
                this->loocv_path_predictions_prefix = "predictions";
            }

            if (this->loocv_path_trees_prefix.empty()){
                std::cout << "loocv_path_trees_prefix is not set or empty, using default 'trees'." << std::endl;
                this->loocv_path_trees_prefix = "trees";
            }

            if (this->use_unique_loocv_path < 0){
                std::cout << "use_unique_loocv_path is not set" << std::endl;
                return false;
            }
        }

        if (this->store_predictionimages == -1){
            std::cout << "store_predictionimages is not set" << std::endl;
            return false;
        } else if (this->store_predictionimages == 1){
            if (this->path_predictionimages.empty()){
                std::cout << "path_predictionimages is not set, although you want to store them ;-)" << std::endl;
                return false;
            }
        }

        if (this->store_dataset == -1){
            std::cout << "store_dataset is not set" << std::endl;
            return false;
        }

        if (this->load_dataset == -1){
            std::cout << "load_dataset is not set" << std::endl;
            return false;
        }

        if (this->path_fixedDataset.empty()){
            std::cout << "path_fixedDataset is not set" << std::endl;
            return false;
        }

        return true;
    }

    inline bool ValidateCellDetectionRegression()
    {
        if (this->path_traindata.empty()){
            std::cout << "path_traindata is empty" << std::endl;
            return false;
        }

        if (this->path_trainlabels.empty()) {
            std::cout << "path_trainlabels is empty" << std::endl;
            return false;
        }

        // ratio of additional background patches
        // use default or the value specified

        // threshold_fg_bg validation
        if (this->threshold_fg_bg > 1) {
            std::cout << "threshold_fg_bg must not be >1" << std::endl;
            return false;
        }

        if (this->path_testdata.empty()) {
            std::cout << "path_testdata is empty" << std::endl;
            return false;
        }

        if (this->path_testlabels.empty()) {
            std::cout << "path_testlabels is empty" << std::endl;
            return false;
        }

        return true;
    }
};

#endif // APPCONTEXTCELLDETREGR_H
