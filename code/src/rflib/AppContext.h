/*
 * AppContext.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef APPCONTEXT_H_
#define APPCONTEXT_H_

#include <ostream>
#include <string>
#include <vector>
#include <libconfig.h++>
#include "icgrf.h"

using namespace std;
using namespace libconfig;



namespace RF_METHOD
{
	enum Enum
	{
		STDRF									= 0,
        NUM = 1,
        NOTSET = 2
	};
}

namespace SPLITFUNCTION_TYPE
{
	enum Enum
	{
		FEATURE_THRESHOLD						= 0,
		OBLIQUE									= 1,
		ORDINAL									= 2,
		PIXELPAIRTEST							= 3,
		HAAR_LIKE								= 4,
		PIXELPAIRTESTCONDITIONED 				= 5,
		SINGLEPIXELTEST							= 6,
		NUM = 7,
		NOTSET = 8
	};
}

namespace SPLITEVALUATION_TYPE_CLASSIFICATION
{
	enum Enum
	{
		ENTROPY									= 0,
		GINI									= 1,
		LOSS_SPECIFIC							= 2,
		NUM = 3,
		NOTSET = 4
	};
}

namespace SPLITEVALUATION_TYPE_REGRESSION
{
	enum Enum
	{
		REDUCTION_IN_VARIANCE 					= 0,
		DIFF_ENTROPY_GAUSS 						= 1,
		DIFF_ENTROPY_GAUSS_BLOCK_POSE			= 2,
		NUM = 3,
		NOTSET = 4
	};
}


namespace LEAFNODE_REGRESSION_TYPE
{
	enum Enum
	{
		ALL 									= 0,
		MEAN 									= 1,
		HILLCLIMB 								= 2,
		MEANSHIFT 								= 3,
		NUM = 4,
		NOTSET = 5
	};
}



namespace REGRESSION_VOTE_ACCUMULATION_METHOD
{
	enum Enum
	{
		ACCUM_MEAN 								= 0,
		ACCUM_HILLCLIMB 						= 1,
		NUM = 2,
		NOTSET = 3
	};
}

namespace NMS_TYPE
{
	enum Enum
	{
		NMS_GALL 								= 0,
		NMS_ADAPT_KERNEL 						= 1,
		NMS_PYRAMID 							= 2,
		NUM = 3,
		NOTSET = 4
	};
}



class AppContext
{
protected:

	// read the config file
	void ReadCore(const std::string& confFile);

	// set default parameters???
	virtual void SetDefaultValues();

	// This method checks for invalid or missing settings (has to be implemented in derivatives!)
	virtual void ValidateHyperparameters() = 0;

	bool ValidateCompleteGeneralPart();

	bool ValidateStandardForestSettings();

	bool ValidateImageSplitFunction();

	bool ValidateADFParameters();

	bool ValidateARFParameters();

	bool ValidateMLDataset();

    bool ValidateCellDetectionRegression();

public:

	// constructors & destructors
	AppContext() {}
	virtual ~AppContext() {}

	// load config file and validate parameters
	virtual void Read(const std::string& confFile)
	{
		this->SetDefaultValues();
		this->ReadCore(confFile);
		this->ValidateHyperparameters();
	}



	// =========================================================================================
	// MEMBER VARIABLES (from other sources, e.g., data, which are important for the appcontext
	// 					 in the splitfunctions and leafnode statistics)
	int num_classes;
	int num_target_variables;
	int num_feature_channels;



    // ######### GENERAL STUFF ######################################
    // ##############################################################
    // method: 0=Standard HF, 1=BT-HF, 2=AD-HF, 3=AGL-HF, 4=DENSE-HF
    RF_METHOD::Enum method;
    // mode: 0=train, 1=predict, 2=train&predict
    int mode;
    // quiet mode
    int quiet;
    // debug on or not
    int debug_on;
    // output file (results, infos, ...)
    string path_output;

    // ######### FOREST STUFF #######################################
    // ##############################################################
    // number of trees
    int num_trees;
    // maximal tree depth
    int max_tree_depth;
    // number of random node tests
    int num_node_tests;
    // number of random node thresholds
    int num_node_thresholds;
    // minimum number of samples required for further splitting
    int min_split_samples;
    // bagging type
	TREE_BAGGING_TYPE::Enum bagging_type;
	// refinement step (in case the <sampling_method> has some out-of-bag samples)
	int do_tree_refinement;
	// offset for storing the trees
	int store_tree_offset;


    // type of split function
    SPLITFUNCTION_TYPE::Enum split_function_type;
	// for OBLIQUE splitting function
	int splitfunction_oblique_k;
    // for ORDINAL splitting function
    int ordinal_split_k;
    // feat channel selection process: 0=same-ch, 1=rand-ch, 2=50/50 chance of same or rand ch
	int split_channel_selection;

	// random subsampling for splitting? 0 if all data is used; >0 is the number of samples per class to use
    int num_random_samples_for_splitting;

    // classification loss type for find best splits (0=std-entropy, 1=gini, 2=loss1, 3=loss2, ...
    SPLITEVALUATION_TYPE_CLASSIFICATION::Enum splitevaluation_type_classification;

	// type of split evaluation for regression nodes: 0=ReductionInVariance, 1=DiffEntropyGauss, ...
    SPLITEVALUATION_TYPE_REGRESSION::Enum splitevaluation_type_regression;

	// DENSE HF ... currently! TODO: make a proper implementation of this such that it can be used everywhere
	LEAFNODE_REGRESSION_TYPE::Enum leafnode_regression_type;


    // HOUGH FOREST SPECIFIC STUFF
    // mean-shifted votes: use max. k votes (k has to be > 0!))
    int mean_shift_votes_k;
    // depth where only regression is allowed
    int depth_regression_only;


    // RIGID DETECTOR SPECIFIC STUFF
    vector<int> bbox_size;
    int do_structured_prediction;



    // TODO: obsolete ???
    // scale the distances of the regression node splits
    int scale_regression_distances;
    // type of regression split node mode search: 0=mean, 1=mode(Kontsch)
    int regression_split_mode_calculation;
    // final discriminative voting weight updates according to final loss function!
    int final_discr_weight_update;


    // ADF specific stuff ...
    int do_classification_weight_updates;
    // weight-update-method for classification
    ADF_LOSS_CLASSIFICATION::Enum global_loss_classification;
    // on/off switch for regression weight-updates!
    int do_regression_weight_updates;
    // weight-update-method for regression
    ADF_LOSS_REGRESSION::Enum global_loss_regression;
    // Delta parameter for the Huber-Regression-Loss
    double global_loss_regression_Huber_delta;
    // shrinkage factor
	double shrinkage;

    // Vote Accumulation Method
    REGRESSION_VOTE_ACCUMULATION_METHOD::Enum regression_vote_accum_method;


    // ADMILF specific stuff
    double mil_positive_bag_ratio;
    int mil_start_depth;
    // labelswitch-type: 0 = DetAnn-like dicing, 1 = Structured-Graphcut
    int mil_label_switch_type;


    // Hough Forest - Test specific stuff
    std::vector<int> patch_size; // the window size to be cropped
    int use_min_max_filters;
    int voting_grid_distance;
    // use meanshift voting during testing (0/1)
    int use_meanshift_voting;
    int print_hough_maps;
    int houghmaps_outputscale;
    int return_bboxes;
    vector<double> avg_bbox_scaling;
    int backproj_bbox_estimation;
    int backproj_bbox_kernel_w;
    double backproj_bbox_cumsum_th;
    int print_detection_images;
    // type of NMS
    NMS_TYPE::Enum nms_type;
    int hough_gauss_kernel;
    int max_detections_per_image;
    double min_detection_peak_height;
    int use_scale_interpolation;
    double poseestim_maxvar_vote;
    double poseestim_min_fgprob_vote;



    // ######### DATA STUFF (Machine Learning data) #################
	// ##############################################################
    string path_traindata;
	string path_trainlabels;
	string path_testdata;
    string path_testlabels;
	// output paths
	string path_trees;
	string path_sampleweight_progress;
	// data-specific stuff
	double traintest_split_ratio;
	int traintest_split_save;
	int traintest_split_load;
	string traintest_split_datapath;
	double general_scaling_factor;


    // ######### DATA STUFF (Vision data) ###########################
    // ##############################################################
    // Train images
    string path_posImages;
    string path_posAnnofile;
    int numPosPatchesPerImage;
    int use_segmasks;
    string path_negImages;
    string path_negAnnofile;
    int numNegPatchesPerImage;
    // Test images
    string path_testImages;
    string path_testFilenames;
    vector<double> test_scales;
    // Dataset storage
    int store_dataset;
    int load_dataset;
    string path_fixedDataset;
    // Output paths
    // string path_trees; // already defined above
    string path_houghimages;
    string path_bboxes;
    string path_detectionimages;
    string path_poseestimates;

    // MIL-specific stuff
    string path_finalsamplelabeling;
    string path_trainprobmaps;

    // stuff for ADFs
    // string path_sampleweight_progress; // already defined above

    // a list of split functions, a split node can randomly choose from
    std::vector<int> split_function_type_list;
    int use_random_split_function;

    // Cell detection stuff
    double ratio_additional_bg_patches; // a ratio to which extent additional background patches will be sampled from an image
    double threshold_fg_bg; // a threshold to be applied to the distance transform images
    vector<int> image_feature_list; // a list of integer values, defining the features to be computed for an image
    int store_predictionimages; // flag whether to store the prediction images to the specified path
    string path_predictionimages; // a path where the prediction images are stored
    // detection images are stored in path_detectionimages

    // post-processing parameters for the cell detection
    int nms_gauss_kernel_w; // width of the gaussian kernel smoothing before nms
    double nms_gauss_kernel_sigma; // sigma of the gaussian kernel

    int nms_kernel_w; // the width of the neighbourhood for local max search
    int max_radius_detection;// a maximum radius around a detection, where a ground truth must lie to be TP

    // the root path of the leave-one-out cross validation
    // directory structure will be the same as in 'bindata'
    int use_unique_loocv_path;
    string path_loocv;
    string loocv_path_predictions_prefix;
    string loocv_path_detections_prefix;
    string loocv_path_trees_prefix;

    // settings for border extension
    int extend_border;
    int border_type;


    friend std::ostream& operator<< (std::ostream& o, const AppContext& hp)
    {
        o << "===========================================================" << endl
          << "method                   = " << hp.method                    << endl
          << "mode                     = " << hp.mode                      << endl
          << "quiet                    = " << hp.quiet                     << endl
          << "path_output              = " << hp.path_output               << endl << endl
          << "num_trees                = " << hp.num_trees                 << endl
          << "max_tree_depth           = " << hp.max_tree_depth            << endl
          << "min_split_samples        = " << hp.min_split_samples         << endl
          << "num_node_tests           = " << hp.num_node_tests            << endl
          << "num_node_thresholds      = " << hp.num_node_thresholds       << endl << endl;
        o << "TODO: write all parameters here" << endl;
        o << "===========================================================" << endl;

        return o;
    }
};

#endif /* APPCONTEXTJOINTCLASSREGR_H_ */
