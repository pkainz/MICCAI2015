/*
 * main.cpp
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#include <iostream>
#include <string>
#include <fstream>
#include <ostream>
#include <eigen3/Eigen/Core>
#include <vector>
#include <stdexcept>

#include <boost/lexical_cast.hpp>

#include "AppContextCellDetRegr.h"
#include "DataLoaderCellDetRegr.h"
#include "SampleImgPatch.h"
#include "LabelMLRegr.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorMLRegr.h"
#include "LeafNodeStatisticsMLRegr.h"
#include "CellDetectorRegr.h"

#include "icgrf.h"

using namespace std;

// typedefs for easier use later
// define the template for the image patch (src is uchar (byte), integral is float))
typedef SplitFunctionImgPatch<uchar, float, AppContextCellDetRegr> TSplitFunctionImgPatch;
typedef RandomForest<SampleImgPatch, LabelMLRegr, TSplitFunctionImgPatch, SplitEvaluatorMLRegr<SampleImgPatch, AppContextCellDetRegr>, LeafNodeStatisticsMLRegr<SampleImgPatch, AppContextCellDetRegr>, AppContextCellDetRegr> TRegressionForest;
typedef Node<SampleImgPatch, LabelMLRegr, TSplitFunctionImgPatch, LeafNodeStatisticsMLRegr<SampleImgPatch, AppContextCellDetRegr>, AppContextCellDetRegr> TNode;

// load the forest and return a pointer to the instance
TRegressionForest* loadForest(AppContextCellDetRegr* apphp){
    // Set the forest parameters
    RFCoreParameters* rfparams = new RFCoreParameters();
    rfparams->m_debug_on = apphp->debug_on;
    rfparams->m_quiet = apphp->quiet;
    rfparams->m_max_tree_depth = apphp->max_tree_depth;
    rfparams->m_min_split_samples = apphp->min_split_samples;
    rfparams->m_num_node_tests = apphp->num_node_tests;
    rfparams->m_num_node_thresholds = apphp->num_node_thresholds;
    rfparams->m_num_random_samples_for_splitting = apphp->num_random_samples_for_splitting;
    rfparams->m_num_trees = apphp->num_trees;
    rfparams->m_bagging_method = (TREE_BAGGING_TYPE::Enum)apphp->bagging_type;
    if (!apphp->quiet)
        cout << (*rfparams) << endl;

    // fix the number of classes to 1
    apphp->num_classes = 1;
    // set univariate target
    apphp->num_target_variables = 1;

    // Loading the forest
    if (!apphp->quiet)
        std::cout << "Loading the forest ... ";
    TRegressionForest* rf;
    switch (apphp->method)
    {
    case RF_METHOD::STDRF:
        rf = new TRegressionForest(rfparams, apphp);
        break;
    default:
        throw std::runtime_error("main.cpp: unknown rf-method defined, must be 0!");
    }
    rf->Load(apphp->path_trees);
    if (!apphp->quiet){
        std::cout << " done." << std::endl;
    }
    return rf;
}

// train a forest on data specified in apphp
void train(AppContextCellDetRegr* apphp, std::vector<boost::filesystem::path>& trainImgFilenames)
{
    // 0) some initial stuff (scaling)
    for (size_t i = 0; i < apphp->patch_size.size(); i++)
        apphp->patch_size[i] = (int)((double)apphp->patch_size[i] * apphp->general_scaling_factor);

    // 1) read the data
    if (!apphp->quiet)
        cout << "Load training data ..." << endl;

    DataLoaderCellDetRegr mydataloader(apphp);
    DataSet<SampleImgPatch, LabelMLRegr> dataset_train = mydataloader.LoadTrainData(trainImgFilenames);

    int num_train_samples, num_classes, num_target_variables, num_feature_channels;
    mydataloader.GetTrainDataProperties(num_train_samples, num_classes, num_target_variables, num_feature_channels);
    // set some information for splitfunctions etc. in the app-context
    apphp->num_classes = num_classes;
    apphp->num_target_variables = num_target_variables;
    apphp->num_feature_channels = num_feature_channels;

    if (!apphp->quiet)
    {
        std::cout << "Dataset information: " << std::endl;
        std::cout << num_train_samples << " patches, ";
        std::cout << num_target_variables << "-d regression targets" << std::endl;
    }

    // 2) prepare the forest - fill the random forest core settings
    RFCoreParameters* rfparams = new RFCoreParameters();
    rfparams->m_debug_on = apphp->debug_on;
    rfparams->m_quiet = apphp->quiet;
    rfparams->m_max_tree_depth = apphp->max_tree_depth;
    rfparams->m_min_split_samples = apphp->min_split_samples;
    rfparams->m_num_node_tests = apphp->num_node_tests;
    rfparams->m_num_node_thresholds = apphp->num_node_thresholds;
    rfparams->m_num_random_samples_for_splitting = apphp->num_random_samples_for_splitting;
    rfparams->m_num_trees = apphp->num_trees;
    rfparams->m_bagging_method = (TREE_BAGGING_TYPE::Enum)apphp->bagging_type;
    if (!apphp->quiet)
        cout << (*rfparams) << endl;

    // 3) train the forest
    TRegressionForest* rf;
    switch (apphp->method)
    {
    case RF_METHOD::STDRF:
        rf = new TRegressionForest(rfparams, apphp);
        break;
    default:
        throw std::runtime_error("main.cpp: unknown rf-method defined, must be 0!");
    }
    if (!apphp->quiet)
        std::cout << "Training ... " << std::endl << std::flush;
    rf->Train(dataset_train);
    if (!apphp->quiet)
        cout << "done" << endl << flush;

    boost::filesystem::path path_trees(apphp->path_trees);
    if (!boost::filesystem::exists(path_trees)){
        if(boost::filesystem::create_directories(path_trees)) {
            if (!apphp->quiet)
                cout << "Created 'trees' directory in '" << path_trees << "'." << endl;
        } else {
            if (!apphp->quiet)
                cout << "Could not create 'trees' directory!" << endl;
        }
    }

    // 4.1) and save results
    rf->Save(apphp->path_trees);

    // 4.2) delete the training samples
    dataset_train.DeleteAllSamples();

    {
        // clear the static vector in the data loader and free the memory
        cout << "Clearing image_feature_data vector... ";
        DataLoaderCellDetRegr::image_feature_data.clear();
        std::vector<std::vector<cv::Mat> > tmp;
        tmp.swap(DataLoaderCellDetRegr::image_feature_data);
        cout << "done." << endl;
    }

    // delete the RF
    delete(rfparams);
    delete(rf);
}

// load a trained forest to predict the test images
void predictTestImages(AppContextCellDetRegr* apphp){

    // 0) some initial stuff (scaling)
    for (size_t i = 0; i < apphp->patch_size.size(); i++)
        apphp->patch_size[i] = (int)((double)apphp->patch_size[i] * apphp->general_scaling_factor);


    boost::filesystem::path path_pred(apphp->path_predictionimages);
    if (!boost::filesystem::exists(path_pred)){
        if(boost::filesystem::create_directories(path_pred)) {
            if (!apphp->quiet)
                cout << "Created 'predictions' directory in '" << path_pred << "'." << endl;
        } else {
            if (!apphp->quiet)
                cout << "Could not create 'predictions' directory!" << endl;
        }
    }


    TRegressionForest* rf;
    rf = loadForest(apphp);

    // let the detector run over the image and produce a distance transform of the estimated
    // cell centers
    CellDetectorRegr detector(rf,apphp);
    detector.PredictTestImages();

    // delete the rf
    delete(rf);
}

// analyze the leaf nodes in the forest
void analyze_leaves(AppContextCellDetRegr* apphp)
{
    TRegressionForest* rf;
    rf = loadForest(apphp);

    // 5) analyse the trees
    std::vector<std::vector<TNode* > > all_leafs;
    all_leafs = rf->GetAllLeafNodes();
    for (size_t t = 0; t < all_leafs.size(); t++)
    {
        for (size_t i = 0; i < all_leafs[t].size(); i++)
        {
            cout << "Tree " << t << ", node " << i << "/" << all_leafs[t].size() << " in depth " << all_leafs[t][i]->m_depth << ":" << endl;
            all_leafs[t][i]->m_leafstats->Print();
        }
    }
}

// analyze the split nodes
void analyze_splitnodes(AppContextCellDetRegr* apphp)
{
    TRegressionForest* rf;
    rf = loadForest(apphp);

    // analyse the trees and take split nodes
    std::vector<std::vector<TNode* > > all_internalnodes;
    all_internalnodes = rf->GetAllInternalNodes();
    cout << "tree node depth ch1 ch2 p1.x p1.y p2.x p2.y re1.x re1.y re1.width re1.height re2.x re2.y re2.width re2.height pxs.size th splitfctn" << endl;
    for (size_t t = 0; t < all_internalnodes.size(); t++)
    {
        for (size_t i = 0; i < all_internalnodes[t].size(); i++)
        {
            //cout << "Tree " << t << ", node " << i << "/" << all_internalnodes[t].size() << " in depth " << all_internalnodes[t][i]->m_depth << ":" << endl;
            cout << t << " " << i << " " << all_internalnodes[t][i]->m_depth << " ";
            all_internalnodes[t][i]->m_splitfunction->Print();
            cout << endl;
        }
    }
}

/**
 * Perform the leave one out cross validation.
 *
 * @param apphp
 */
void loocv(AppContextCellDetRegr* apphp){
    // train the forest on N-1 images
    // 1) load foreground and target image data
    // read in all file names from the training data directory
    vector<boost::filesystem::path> trainImgFilenames;
    vector<boost::filesystem::path> allImgFilenames;
    // list all image names in the train directory
    ListAllTrainImgFilenames(apphp, allImgFilenames);

    // the path to the validation image within the train directory
    boost::filesystem::path valImgFilename;

    string loocv_root;
    // make the cross validation directory structure
    if (apphp->use_unique_loocv_path){
        std::time_t t = std::time(NULL);
        char date_str[20];
        std::strftime(date_str, 20, "%Y-%m-%d_%H%M%S", std::localtime(&t));

        loocv_root = apphp->path_loocv + "/" + date_str;
    }else{
        loocv_root = apphp->path_loocv;
    }

    boost::filesystem::path path_loocv(loocv_root);
    if (!boost::filesystem::exists(path_loocv)){
        if(boost::filesystem::create_directories(path_loocv)) {
            if (!apphp->quiet)
                cout << "Created 'loocv' root directory in '" << loocv_root << "'." << endl;
        } else {
            if (!apphp->quiet)
                cout << "Could not create 'loocv' root directory!" << endl;
        }
    }

    // span a for loop around these data
    for (int cross_val_idx = 0; cross_val_idx < allImgFilenames.size(); cross_val_idx++){

        std::vector<string> dirs;
        dirs.push_back(loocv_root + "/" + boost::lexical_cast<string>(cross_val_idx) + "/" + apphp->loocv_path_trees_prefix);
        dirs.push_back(loocv_root + "/" + boost::lexical_cast<string>(cross_val_idx) + "/" + apphp->loocv_path_predictions_prefix);
        //dirs.push_back(loocv_root + "/" + boost::lexical_cast<string>(cross_val_idx) + "/" + apphp->loocv_path_detections_prefix);

        // make the directories
        for (int i = 0; i < dirs.size(); i++){
            boost::filesystem::path tmp(dirs[i]);
            if (!boost::filesystem::exists(tmp)){
                if (boost::filesystem::create_directories(tmp)){
                    if (!apphp->quiet)
                        cout << "Created directory '" << tmp.c_str() << "'." << endl;
                } else {
                    if (!apphp->quiet)
                        cout << "Could not create directory '" << tmp.c_str() << "'!" << endl;
                }
            }
        }

        // change some directories to the new locations
        // in order to redirect the output to the correct location
        apphp->path_trees = dirs[0] + "/";
        apphp->path_predictionimages = dirs[1] + "/";
        apphp->path_detectionimages = dirs[2] + "/";

        // redirect the path to the fixed dataset
        apphp->path_fixedDataset = loocv_root + "/" +
                boost::lexical_cast<string>(cross_val_idx) + "/fixed_dataset.txt";

        cout << endl << "Running leave-one-out cross validation ";
        cout << (cross_val_idx+1) << " of " << allImgFilenames.size() << endl;

        valImgFilename = allImgFilenames[cross_val_idx];
        trainImgFilenames.clear();
        trainImgFilenames = allImgFilenames;
        // remove the index from the train images
        trainImgFilenames.erase(trainImgFilenames.begin() + cross_val_idx);
        if (!apphp->quiet){
            cout << "training on n=" << trainImgFilenames.size() << " images" << endl;
            cout << "validation image: " << valImgFilename.filename().c_str() << endl;
        }

        // train the random forest model
        train(apphp, trainImgFilenames);

        // predict the validation image
        // let the detector run over the image and produce a distance transform of the estimated
        // cell centers
        TRegressionForest* rf;
        rf = loadForest(apphp);

        CellDetectorRegr detector(rf,apphp);
        detector.PredictSingleImage(valImgFilename);

        delete(rf);
    }
}


int main(int argc, char* argv[])
{

    // 1) read input arguments
    std::string path_configfile;
    if (argc < 2)
    {
        std::cout << "Specify a config file" << std::endl;
        exit(-1);
    }
    else
    {
        path_configfile = argv[1];
    }


    // 2) read configuration
    AppContextCellDetRegr apphp; // regression detection
    apphp.Read(path_configfile);

    if (!apphp.quiet){
        std::cout << "Parsed configuration file from " << path_configfile << std::endl;
        std::cout << apphp;
        std::cout << std::endl;
    }

    std::vector<boost::filesystem::path> trainImgFilenames;
    //try
    //{
    switch (apphp.mode) // the mode the forest is run in
    {
    case 0:
        ListAllTrainImgFilenames(&apphp, trainImgFilenames);
        train(&apphp, trainImgFilenames);
        break;
    case 1:
        predictTestImages(&apphp);
        break;
    case 2:
        ListAllTrainImgFilenames(&apphp, trainImgFilenames);
        train(&apphp, trainImgFilenames);
        predictTestImages(&apphp);
        break;
    case 3:
        analyze_leaves(&apphp);
        break;
    case 4:
        analyze_splitnodes(&apphp);
        break;
    case 5:
        // each cross validation is writing to a subdirectory
        // otherwise, tree and prediction files get overwritten every time!
        // do a leave-one-out cross-validation
        loocv(&apphp);
        break;
    default:
        throw std::runtime_error("main.cpp: wrong mode specified!");
        break;
    }
    //}
    //catch (std::exception& e)
    //{
    //	std::cout << "ERROR occured, exception thrown:" << std::endl;
    //	std::cout << e.what() << std::endl;
    //}

    if (!apphp.quiet)
        std::cout << "Program should be finished now ..." << std::endl << std::flush;
    return 0;
}
