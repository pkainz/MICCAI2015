/*
 * AppContext.cpp
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef APPCONTEXT_CPP_
#define APPCONTEXT_CPP_

#include "AppContext.h"


void AppContext::ReadCore(const std::string& confFile)
{
    try
    {
        // helper variable
        int tmp = -1;

        // Read the config file
        Config configFile;
        configFile.readFile(confFile.c_str());

        // ###################################################################################
        // ###################################################################################
        // General parameters
        // ###################################################################################
        if (configFile.exists("General.method"))
        {
            tmp = configFile.lookup("General.method");
            if (tmp < 0 || tmp >= RF_METHOD::NUM)
            {
                cout << "Error: wrong model-type specified (out-of-range)" << endl;
                exit(-1);
            }
            this->method = (RF_METHOD::Enum)tmp;
        }

        configFile.lookupValue("General.mode", this->mode);
        configFile.lookupValue("General.quiet", this->quiet);
        configFile.lookupValue("General.debug_on", this->debug_on);
        configFile.lookupValue("General.path_output", this->path_output);




        // ###################################################################################
        // ###################################################################################
        // Forest parameters (general)
        // ###################################################################################
        configFile.lookupValue("Forest.num_trees", this->num_trees);
        configFile.lookupValue("Forest.max_tree_depth", this->max_tree_depth);
        configFile.lookupValue("Forest.min_split_samples", this->min_split_samples);
        configFile.lookupValue("Forest.num_node_tests", this->num_node_tests);
        configFile.lookupValue("Forest.num_node_thresholds", this->num_node_thresholds);

        if (configFile.exists("Forest.bagging_type"))
        {
            tmp = configFile.lookup("Forest.bagging_type");
            if (tmp < 0 || tmp >= TREE_BAGGING_TYPE::NUM)
            {
                cout << "Error: wrong tree bagging-type (out-of-range)" << endl;
                exit(-1);
            }
            this->bagging_type = (TREE_BAGGING_TYPE::Enum)tmp;
        }

        configFile.lookupValue("Forest.do_tree_refinement", this->do_tree_refinement);
        configFile.lookupValue("Forest.store_tree_offset", this->store_tree_offset);

        if (configFile.exists("Forest.splitfunction_type"))
        {
            tmp = configFile.lookup("Forest.splitfunction_type");
            if (tmp < 0 || tmp >= SPLITFUNCTION_TYPE::NUM)
            {
                cout << "Error: wrong split-function-type specified (out-of-range)" << endl;
                exit(-1);
            }
            this->split_function_type = (SPLITFUNCTION_TYPE::Enum)tmp;
        }

        configFile.lookupValue("Forest.ordinal_split_k", this->ordinal_split_k);
        configFile.lookupValue("Forest.splitfunction_oblique_k", this->splitfunction_oblique_k);
        configFile.lookupValue("Forest.num_random_samples_for_splitting", this->num_random_samples_for_splitting);

        if (configFile.exists("Forest.leafnode_regression_type"))
        {
            tmp = configFile.lookup("Forest.leafnode_regression_type");
            if (tmp < 0 || tmp >= LEAFNODE_REGRESSION_TYPE::NUM)
            {
                cout << "ERROR: wrong leafnode_regression_type specified (out-of-range)" << endl;
                exit(-1);
            }
            this->leafnode_regression_type = (LEAFNODE_REGRESSION_TYPE::Enum)tmp;
        }

        if (configFile.exists("Forest.splitevaluation_type_classification"))
        {
            tmp = configFile.lookup("Forest.splitevaluation_type_classification");
            if (tmp < 0 || tmp >= SPLITEVALUATION_TYPE_CLASSIFICATION::NUM)
            {
                cout << "ERROR: wrong splitevaluation_type_classification specified (out-of-range)" << endl;
                exit(-1);
            }
            this->splitevaluation_type_classification = (SPLITEVALUATION_TYPE_CLASSIFICATION::Enum)tmp;
        }

        if (configFile.exists("Forest.splitevaluation_type_regression"))
        {
            tmp = configFile.lookup("Forest.splitevaluation_type_regression");
            if (tmp < 0 || tmp >= SPLITEVALUATION_TYPE_REGRESSION::NUM)
            {
                cout << "ERROR: wrong splitevaluation_type_regression specified (out-of-range)" << endl;
                exit(-1);
            }
            this->splitevaluation_type_regression = (SPLITEVALUATION_TYPE_REGRESSION::Enum)tmp;
        }


        // ###################################################################################
        // ###################################################################################
        // ADF & ARF specific stuff
        // ###################################################################################
        configFile.lookupValue("Forest.do_classification_weight_updates", this->do_classification_weight_updates);

        if (configFile.exists("Forest.global_loss_classification"))
        {
            tmp = configFile.lookup("Forest.global_loss_classification");
            if (tmp < 0 || tmp >= ADF_LOSS_CLASSIFICATION::NUM)
            {
                cout << "Error: wrong <global_loss_classification> specified (out-of-range)" << endl;
                exit(-1);
            }
            this->global_loss_classification = (ADF_LOSS_CLASSIFICATION::Enum)tmp;
        }

        configFile.lookupValue("Forest.do_regression_weight_updates", this->do_regression_weight_updates);

        if (configFile.exists("Forest.global_loss_regression"))
        {
            tmp = configFile.lookup("Forest.global_loss_regression");
            if (tmp < 0 || tmp >= ADF_LOSS_REGRESSION::NUM)
            {
                cout << "Error: wrong <global_loss_regression> specified (out-of-range)" << endl;
                exit(-1);
            }
            this->global_loss_regression = (ADF_LOSS_REGRESSION::Enum)tmp;
        }

        configFile.lookupValue("Forest.global_loss_regression_Huber_delta", this->global_loss_regression_Huber_delta);
        configFile.lookupValue("Forest.shrinkage", this->shrinkage);





        // ###################################################################################
        // ###################################################################################
        // MIL-specific stuff ...
        // ###################################################################################
        configFile.lookupValue("Forest.mil_positive_bag_ratio", this->mil_positive_bag_ratio);
        configFile.lookupValue("Forest.mil_start_depth", this->mil_start_depth);
        configFile.lookupValue("Forest.mil_label_switch_type", this->mil_label_switch_type);




        // ###################################################################################
        // ###################################################################################
        // General Forest / Data stuff
        // ###################################################################################
        configFile.lookupValue("Data.path_trees", this->path_trees);
        configFile.lookupValue("Data.path_sampleweight_progress", this->path_sampleweight_progress);

        // Dataset parameters
        configFile.lookupValue("Data.traintest_split_ratio", this->traintest_split_ratio);
        configFile.lookupValue("Data.traintest_split_save", this->traintest_split_save);
        configFile.lookupValue("Data.traintest_split_load", this->traintest_split_load);
        configFile.lookupValue("Data.traintest_split_datapath", this->traintest_split_datapath);
        configFile.lookupValue("Data.path_traindata", this->path_traindata);
        configFile.lookupValue("Data.path_trainlabels", this->path_trainlabels);
        configFile.lookupValue("Data.path_testdata", this->path_testdata);
        configFile.lookupValue("Data.path_testlabels", this->path_testlabels);
        configFile.lookupValue("Data.general_scaling_factor", this->general_scaling_factor);





        // ###################################################################################
        // ###################################################################################
        // Hough Forest specifc stuff (Forest)
        // ###################################################################################
        configFile.lookupValue("Forest.mean_shift_votes_k", this->mean_shift_votes_k);
        configFile.lookupValue("Forest.depth_regression_only", this->depth_regression_only);
        configFile.lookupValue("Forest.split_channel_selection", this->split_channel_selection);

        configFile.lookupValue("Forest.scale_regression_distances", this->scale_regression_distances);
        configFile.lookupValue("Forest.regression_split_mode_calculation", this->regression_split_mode_calculation);
        configFile.lookupValue("Forest.final_discr_weight_update", this->final_discr_weight_update);

        // OBSOLETE ?!?!?
        if (configFile.exists("Forest.regression_vote_accum_method"))
        {
            tmp = configFile.lookup("Forest.regression_vote_accum_method");
            if (tmp < 0 || tmp >= REGRESSION_VOTE_ACCUMULATION_METHOD::NUM)
            {
                cout << "ERROR: wrong REGRESSION_VOTE_ACCUMULATION_METHOD specified (out-of-range)" << endl;
                exit(-1);
            }
            this->regression_vote_accum_method = (REGRESSION_VOTE_ACCUMULATION_METHOD::Enum)tmp;
        }

        // ###################################################################################
        // ###################################################################################
        // Hough Forest specifc stuff (Testing / General)
        // + Rigid Detector specific stuff (Testing / General)
        // ###################################################################################
        //configFile.lookupValue("Forest.patch_size", this->patch_size);
        if (configFile.exists("Forest.patch_size"))
        {
            for (int i = 0; i < configFile.lookup("Forest.patch_size").getLength(); i++)
                this->patch_size.push_back(configFile.lookup("Forest.patch_size")[i]);
            if (this->patch_size.size() != 2)
            {
                cout << "The patch size should be 2-dimensional!" << endl;
                exit(-1);
            }
        }
        if (configFile.exists("Forest.bbox_size"))
        {
            for (int i = 0; i < configFile.lookup("Forest.bbox_size").getLength(); i++)
                this->bbox_size.push_back(configFile.lookup("Forest.bbox_size")[i]);
            if (this->bbox_size.size() != 2)
            {
                cout << "The bbox size should be 2-dimensional!" << endl;
                exit(-1);
            }
        }
        configFile.lookupValue("Forest.use_min_max_filters", this->use_min_max_filters);
        configFile.lookupValue("Forest.use_meanshift_voting", this->use_meanshift_voting);
        configFile.lookupValue("Forest.voting_grid_distance", this->voting_grid_distance);
        configFile.lookupValue("Forest.print_hough_maps", this->print_hough_maps);
        configFile.lookupValue("Forest.houghmaps_outputscale", this->houghmaps_outputscale);
        configFile.lookupValue("Forest.return_bboxes", this->return_bboxes);

        if (configFile.exists("Forest.avg_bbox_scaling"))
        {
            for (unsigned int i = 0; i < configFile.lookup("Forest.avg_bbox_scaling").getLength(); i++)
                this->avg_bbox_scaling.push_back(configFile.lookup("Forest.avg_bbox_scaling")[i]);
        }

        configFile.lookupValue("Forest.backproj_bbox_estimation", this->backproj_bbox_estimation);
        configFile.lookupValue("Forest.backproj_bbox_kernel_w", this->backproj_bbox_kernel_w);
        configFile.lookupValue("Forest.backproj_bbox_cumsum_th", this->backproj_bbox_cumsum_th);
        configFile.lookupValue("Forest.print_detection_images", this->print_detection_images);

        if (configFile.exists("Forest.nms_type"))
        {
            tmp = configFile.lookup("Forest.nms_type");
            if (tmp < 0 || tmp > NMS_TYPE::NUM)
            {
                cout << "Error: wrong nms-type specified (out-of-range)" << endl;
                exit(-1);
            }
            this->nms_type = (NMS_TYPE::Enum)tmp;
        }

        configFile.lookupValue("Forest.hough_gauss_kernel_w", this->hough_gauss_kernel);
        configFile.lookupValue("Forest.max_detections_per_image", this->max_detections_per_image);
        configFile.lookupValue("Forest.min_detection_peak_height", this->min_detection_peak_height);
        configFile.lookupValue("Forest.use_scale_interpolation", this->use_scale_interpolation);
        configFile.lookupValue("Forest.do_structured_prediction", this->do_structured_prediction);





        // ###################################################################################
        // ###################################################################################
        // Headpose estimation with Hough Forest specifc stuff
        // ###################################################################################
        configFile.lookupValue("Forest.poseestim_maxvar_vote", this->poseestim_maxvar_vote);
        configFile.lookupValue("Forest.poseestim_min_fgprob_vote", this->poseestim_min_fgprob_vote);



        // ###################################################################################
        // ###################################################################################
        // HoughForest / Data (for object detection and head-pose estimation)
        // ###################################################################################
        configFile.lookupValue("Data.posExamplesPath", this->path_posImages);
        configFile.lookupValue("Data.posExamplesFile", this->path_posAnnofile);
        configFile.lookupValue("Data.numPosPatchesPerImage", this->numPosPatchesPerImage);
        configFile.lookupValue("Data.use_segmasks", this->use_segmasks);
        configFile.lookupValue("Data.negExamplesPath", this->path_negImages);
        configFile.lookupValue("Data.negExamplesFile", this->path_negAnnofile);
        configFile.lookupValue("Data.numNegPatchesPerImage", this->numNegPatchesPerImage);
        configFile.lookupValue("Data.testImagesPath", this->path_testImages);
        configFile.lookupValue("Data.testImagesFilenames", this->path_testFilenames);

        if (configFile.exists("Data.test_image_scales"))
        {
            for (unsigned int i = 0; i < configFile.lookup("Data.test_image_scales").getLength(); i++)
            {
                this->test_scales.push_back(configFile.lookup("Data.test_image_scales")[i]);
                if (i > 0)
                {
                    if (test_scales[i-1] >= test_scales[i])
                    {
                        cout << "ERROR: the test scales should be in a strict increasing order!" << endl;
                        exit(-1);
                    }
                }
            }
        }

        // TODO: rename some stuff here, or merge it with the ML parameters!
        configFile.lookupValue("Data.store_dataset", this->store_dataset);
        configFile.lookupValue("Data.load_dataset", this->load_dataset);
        configFile.lookupValue("Data.fixeddatasetpath", this->path_fixedDataset);
        // defined above in the genearl forest / data section
        //configFile.lookupValue("Data.sampleweightprogresspath", this->path_sampleweight_progress);
        configFile.lookupValue("Data.finalsamplelabelingpath", this->path_finalsamplelabeling);
        configFile.lookupValue("Data.trainprobmapspath", this->path_trainprobmaps);

        configFile.lookupValue("Data.houghimgpath", this->path_houghimages);
        configFile.lookupValue("Data.bboxpath", this->path_bboxes);
        configFile.lookupValue("Data.detectionimgpath", this->path_detectionimages);
        configFile.lookupValue("Data.poseestimatespath", this->path_poseestimates);

        // random split function type
        configFile.lookupValue("Forest.use_random_split_function", this->use_random_split_function);

        if (this->use_random_split_function){
            for (int i = 0; i < configFile.lookup("Forest.splitfunction_type_list").getLength(); i++){
                int spfctn_id = configFile.lookup("Forest.splitfunction_type_list")[i];

                if (spfctn_id < 0 || spfctn_id >= SPLITFUNCTION_TYPE::NUM){
                    cout << "ERROR: this splitfunction_type does not exist: " << spfctn_id << endl;
                    exit(-1);
                }
                this->split_function_type_list.push_back(spfctn_id);
            }
        }

        // CELL DETECTION STUFF
        configFile.lookupValue("Data.ratio_additional_bg_patches", this->ratio_additional_bg_patches);
        configFile.lookupValue("Data.threshold_fg_bg", this->threshold_fg_bg);

        // feature list for the forest
        if (configFile.exists("Forest.image_feature_list")){
            for (int i = 0; i < configFile.lookup("Forest.image_feature_list").getLength(); i++)
                this->image_feature_list.push_back(configFile.lookup("Forest.image_feature_list")[i]);

            if (this->image_feature_list.size() < 1)
            {
                cout << "ERROR: The image_feature_list must not be empty!" << endl;
                exit(-1);
            }
        } else {
            cout << "ERROR: The image_feature_list config parameter does not exist!" << endl;
            exit(-1);
        }

        configFile.lookupValue("Data.store_predictionimages", this->store_predictionimages);
        configFile.lookupValue("Data.predictionimgpath", this->path_predictionimages);
        configFile.lookupValue("Forest.max_radius_detection", this->max_radius_detection);

        // post processing
        configFile.lookupValue("Forest.nms_gauss_kernel_w", this->nms_gauss_kernel_w);
        configFile.lookupValue("Forest.nms_gauss_kernel_sigma", this->nms_gauss_kernel_sigma);

        configFile.lookupValue("Forest.nms_kernel_w", this->nms_kernel_w);

        // cross-validation settings
        configFile.lookupValue("Data.use_unique_loocv_path", this->use_unique_loocv_path);
        configFile.lookupValue("Data.path_loocv", this->path_loocv);
        configFile.lookupValue("Data.loocv_path_predictions_prefix", this->loocv_path_predictions_prefix);
        configFile.lookupValue("Data.loocv_path_detections_prefix", this->loocv_path_detections_prefix);
        configFile.lookupValue("Data.loocv_path_trees_prefix", this->loocv_path_trees_prefix);

        // border extension settings
        configFile.lookupValue("Forest.extend_border", this->extend_border);
        configFile.lookupValue("Forest.border_type", this->border_type);

    }
    catch (libconfig::ParseException& e)
    {
        std::cout << "Error reading Configuration file (" << confFile.c_str() << ")!" << std::endl;
        std::cout << "ParseException at Line " << e.getLine() << ": " << e.getError() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (libconfig::SettingException& e)
    {
        std::cout << "Error reading Configuration file (" << confFile.c_str() << ")!" << std::endl;
        std::cout << "SettingException at " << e.getPath() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (std::exception& e)
    {
        std::cout << "Error reading Configuration file (" << confFile.c_str() << ")!" << std::endl;
        std::cout << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cout << "Error reading Configuration file (" << confFile.c_str() << ")!" << std::endl;
        std::cout << "Unknown Exception" << std::endl;
        exit(EXIT_FAILURE);
    }
}


void AppContext::SetDefaultValues()
{
    this->method = RF_METHOD::NOTSET;
    this->mode = -1;
    this->quiet = -1;
    this->debug_on = -1;
    this->path_output = "";

    this->num_trees = -1;
    this->max_tree_depth = -1;
    this->num_node_tests = -1;
    this->num_node_thresholds = -1;
    this->min_split_samples = -1;
    this->bagging_type = TREE_BAGGING_TYPE::NOTSET;
    this->do_tree_refinement = -1;
    this->store_tree_offset = 0; // this is really optional (so if it is not set, then we set it to zero!)

    this->split_function_type = SPLITFUNCTION_TYPE::NOTSET;
    this->splitfunction_oblique_k = -1;
    this->ordinal_split_k = -1;
    this->split_channel_selection = -1;
    this->num_random_samples_for_splitting = -1;

    this->splitevaluation_type_classification = SPLITEVALUATION_TYPE_CLASSIFICATION::NOTSET;
    this->splitevaluation_type_regression = SPLITEVALUATION_TYPE_REGRESSION::NOTSET;
    this->leafnode_regression_type = LEAFNODE_REGRESSION_TYPE::NOTSET;

    this->mean_shift_votes_k = -1;
    this->depth_regression_only = -1;

    this->scale_regression_distances = -1;
    this->regression_split_mode_calculation = -1;
    this->final_discr_weight_update = -1;

    this->do_classification_weight_updates = -1;
    this->global_loss_classification = ADF_LOSS_CLASSIFICATION::NOTSET;
    this->do_regression_weight_updates = -1;
    this->global_loss_regression = ADF_LOSS_REGRESSION::NOTSET;
    this->global_loss_regression_Huber_delta = -1.0;
    this->shrinkage = -1.0;
    this->regression_vote_accum_method = REGRESSION_VOTE_ACCUMULATION_METHOD::NOTSET;

    this->mil_positive_bag_ratio = -1.0;
    this->mil_start_depth = -1;
    this->mil_label_switch_type = -1;

    this->patch_size.clear();
    this->bbox_size.clear();
    this->do_structured_prediction = -1;
    this->use_min_max_filters = -1;
    this->voting_grid_distance = -1;
    this->use_meanshift_voting = -1;
    this->print_hough_maps = -1;
    this->houghmaps_outputscale = -1;
    this->return_bboxes = -1;
    this->avg_bbox_scaling.clear();
    this->backproj_bbox_estimation = -1;
    this->backproj_bbox_kernel_w = -1;
    this->backproj_bbox_cumsum_th = -1.0;
    this->print_detection_images = -1;
    this->nms_type = NMS_TYPE::NOTSET;
    this->hough_gauss_kernel = -1;
    this->max_detections_per_image = -1;
    this->min_detection_peak_height = -1.0;
    this->use_scale_interpolation = -1;
    this->poseestim_maxvar_vote = -1.0;
    this->poseestim_min_fgprob_vote = -1.0;

    this->path_traindata = "";
    this->path_trainlabels = "";
    this->path_testdata = "";
    this->path_testlabels = "";
    this->path_trees = "";
    this->path_sampleweight_progress = "";
    this->traintest_split_ratio = -1;
    this->traintest_split_save = -1;
    this->traintest_split_load = -1;
    this->traintest_split_datapath = "";
    // this is a real default parameter, so if it isn't specified in config, we use 1.0 (no scaling)
    this->general_scaling_factor = 1.0;

    this->path_posImages = "";
    this->path_posAnnofile = "";
    this->numPosPatchesPerImage = -1;
    this->use_segmasks = -1;
    this->path_negImages = "";
    this->path_negAnnofile = "";
    this->numNegPatchesPerImage = -1;
    this->path_testImages = "";
    this->path_testFilenames = "";
    this->test_scales.clear();
    this->store_dataset = -1;
    this->load_dataset = -1;
    this->path_fixedDataset = "";
    this->path_houghimages = "";
    this->path_bboxes = "";
    this->path_detectionimages = "";
    this->path_poseestimates = "";

    this->path_finalsamplelabeling = "";
    this->path_trainprobmaps = "";

    // random split function type
    this->use_random_split_function = -1;
    this->split_function_type_list.clear();

    // CELL DETECTION specific stuff
    // a parameter for randomly sampling the background patches
    this->ratio_additional_bg_patches = 0.5;
    this->threshold_fg_bg = 0.0;
    this->image_feature_list.clear();
    this->store_predictionimages = -1;
    this->path_predictionimages = "";

    // nms settings
    this->nms_gauss_kernel_w = -1;
    this->nms_gauss_kernel_sigma = -1.0;
    this->nms_kernel_w = -1;

    // detection settinga
    this->max_radius_detection = -1;

    this->use_unique_loocv_path = -1;
    this->path_loocv = "";
    this->loocv_path_predictions_prefix = "";
    this->loocv_path_detections_prefix = "";
    this->loocv_path_trees_prefix = "";

    // border extension settings
    this->extend_border = -1;
    this->border_type = cv::BORDER_DEFAULT;
}


bool AppContext::ValidateCompleteGeneralPart()
{
    if (this->method == RF_METHOD::NOTSET)
        return false;

    if (this->mode == -1)
        return false;

    if (this->quiet == -1)
        return false;

    if (this->debug_on == -1)
        return false;

    if (this->path_output.empty())
        return false;

    return true;
}


bool AppContext::ValidateStandardForestSettings()
{
    if (this->num_trees == -1)
        return false;

    if (this->max_tree_depth == -1)
        return false;

    if (this->min_split_samples == -1)
        return false;

    if (this->num_node_tests == -1)
        return false;

    if (this->num_node_thresholds == -1)
        return false;

    if (this->num_random_samples_for_splitting == -1)
        return false;

    if (this->bagging_type == TREE_BAGGING_TYPE::NOTSET)
        return false;

    if (this->do_tree_refinement == -1)
        return false;

    if (this->use_random_split_function == 1){
        if (this->split_function_type_list.empty()){
            std::cout << "The splitfunction_type_list is empty or not set" << std::endl;
            return false;
        }
    } else {
        if (this->split_function_type == SPLITFUNCTION_TYPE::NOTSET)
            return false;

        if (this->split_function_type == SPLITFUNCTION_TYPE::OBLIQUE)
            if (this->splitfunction_oblique_k == -1)
                return false;

        if (this->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
            if (this->ordinal_split_k == -1)
                return false;
    }

    if (this->splitevaluation_type_classification == SPLITEVALUATION_TYPE_CLASSIFICATION::NOTSET && this->splitevaluation_type_regression == SPLITEVALUATION_TYPE_REGRESSION::NOTSET)
    {
        std::cout << "Either classification or regression splitevaluation type has to be set!" << std::endl;
        return false;
    }

    if (this->path_trees.empty())
        return false;

    return true;
}


bool AppContext::ValidateImageSplitFunction()
{
    bool split_ok = true;

    if (!this->split_function_type == SPLITFUNCTION_TYPE::PIXELPAIRTEST &&
            !this->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE &&
            !this->split_function_type == SPLITFUNCTION_TYPE::PIXELPAIRTESTCONDITIONED &&
            !this->split_function_type == SPLITFUNCTION_TYPE::SINGLEPIXELTEST &&
            !this->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
        split_ok = false;

    if (this->split_channel_selection == -1)
        split_ok = false;

    return split_ok;
}


bool AppContext::ValidateADFParameters()
{
    if (this->do_classification_weight_updates == -1)
        return false;

    if (this->global_loss_classification == ADF_LOSS_CLASSIFICATION::NOTSET)
        return false;

    if (this->shrinkage < 0.0)
        return false;

    return true;
}


bool AppContext::ValidateARFParameters()
{
    if (this->do_regression_weight_updates == -1)
        return false;

    if (this->global_loss_regression == ADF_LOSS_REGRESSION::NOTSET)
        return false;

    if (this->global_loss_regression_Huber_delta < 0.0)
        return false;

    if (this->shrinkage < 0.0)
        return false;

    return true;
}


bool AppContext::ValidateMLDataset()
{
    if (this->path_traindata.empty())
        return false;

    if (this->path_trainlabels.empty())
        return false;

    if (this->path_testdata.empty() || this->path_testlabels.empty())
    {
        if (this->traintest_split_ratio < 0.0)
            return false;

        if (this->traintest_split_load == -1)
            return false;

        if (this->traintest_split_save == -1)
            return false;

        if (this->traintest_split_datapath.empty())
            return false;
    }

    return true;
}

#endif /* APPCONTEXTJOINTCLASSREGR_H_ */
