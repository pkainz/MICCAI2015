/*
 * RFCoreParameters.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef RFCOREPARAMETERS_H_
#define RFCOREPARAMETERS_H_



namespace TREE_BAGGING_TYPE
{
    enum Enum
    {
    	NONE								= 0,
        SUBSAMPLE_WITH_REPLACEMENT 			= 1,
        FIXED_RANDOM_SUBSET					= 2,
        NUM = 3,
        NOTSET = 4
    };
}

namespace ADF_LOSS_CLASSIFICATION
{
	enum Enum
	{
		// Classification losses
		GRAD_LOGIT 						= 0,
		GRAD_HINGE 						= 1,
		GRAD_SAVAGE 					= 2,
		GRAD_EXP 						= 3,
		GRAD_TANGENT 					= 4,
		ZERO_ONE 						= 5,
		NUM = 6,
		NOTSET = 7
	};
}

namespace ADF_LOSS_REGRESSION
{
	enum Enum
	{
		SQUARED_LOSS 					= 0,
		ABSOLUTE_LOSS 					= 1,
		HUBER_LOSS 						= 2,
		NUM = 3,
		NOTSET = 4
	};
}

namespace ADF_LOSS_MIL
{
	enum Enum
	{
		NOISY_OR 						= 0,
		NOISY_OR_GM 					= 1,
		NOISY_OR_PRIOR 					= 2,
		NUM = 3,
		NOTSET = 4
	};
}

class RFCoreParameters
{
private:

	// set default values
	void SetDefaultValues()
	{
		this->m_quiet = 0;
		this->m_debug_on = 0;

		this->m_num_trees = 4;
		this->m_max_tree_depth = 12;
		this->m_min_split_samples = 10;
		this->m_num_node_tests = 100;
		this->m_num_node_thresholds = 10;
		this->m_bagging_method = TREE_BAGGING_TYPE::NONE;
		this->m_do_tree_refinement = 1;
		this->m_num_random_samples_for_splitting = 0;

		this->m_adf_loss_classification = ADF_LOSS_CLASSIFICATION::GRAD_LOGIT;
		this->m_adf_loss_regression = ADF_LOSS_REGRESSION::SQUARED_LOSS;
		this->m_Huberloss_delta = 0.4;
		this->m_adf_loss_mil = ADF_LOSS_MIL::NOISY_OR;
		this->m_shrinkage = 1.0;
	}

public:

	RFCoreParameters()
	{
		this->SetDefaultValues();
	};

	~RFCoreParameters()
	{
	}



	// MEMBER VARIABLES

	// -------------------------------------------------------------------------
	// Verbosity settings
	// if set to "1", no output is produced by the core
	int m_quiet;
	// if set to "1", debug messages are printed by the core
	int m_debug_on;


	// -------------------------------------------------------------------------
	// General forest settings
	int m_num_trees;
	int m_max_tree_depth;
	// - minimum number of samples required for further splitting
	int m_min_split_samples;
	int m_num_node_tests;
    int m_num_node_thresholds;
    // - bagging method
    TREE_BAGGING_TYPE::Enum m_bagging_method;
    // - refinement step (in case the <sampling_method> has some out-of-bag samples)
    int m_do_tree_refinement;
    // - random subsampling for splitting? set to "0" if all data should be used
    //    set to ">0" if subsampling should be active, the number provided is the number samples used for
    //                for optimization (randomly chosen)
    int m_num_random_samples_for_splitting;

    // ADForest specific stuff
    // - weight-update-method
    ADF_LOSS_CLASSIFICATION::Enum m_adf_loss_classification;
    ADF_LOSS_REGRESSION::Enum m_adf_loss_regression;
    ADF_LOSS_MIL::Enum m_adf_loss_mil;
    // - loss-specific parameters
    double m_Huberloss_delta;
    // - shrinkage factor
	double m_shrinkage;




    // This method just checks for invalid settings
    void ValidateHyperparameters()
    {
        // TODO: implement this ...
    }


    // Method for printing the Hyperparameters
    friend std::ostream& operator<< (std::ostream &o, const RFCoreParameters &hp)
    {
    	o << "===========================================================" << std::endl;
    	o << "Random Forest Core Parameteres" << std::endl;
        o << "===========================================================" << std::endl;

        o << "quiet                    = " << hp.m_quiet                   << std::endl;
        o << "debug_on                 = " << hp.m_debug_on				   << std::endl << std::endl;

        o << "num_trees                = " << hp.m_num_trees                 << std::endl;
        o << "max_tree_depth           = " << hp.m_max_tree_depth            << std::endl;
        o << "min_split_samples        = " << hp.m_min_split_samples         << std::endl;
        o << "num_node_tests           = " << hp.m_num_node_tests            << std::endl;
        o << "num_node_thresholds      = " << hp.m_num_node_thresholds       << std::endl;
        switch (hp.m_bagging_method)
        {
        case TREE_BAGGING_TYPE::NONE:
        	o << "bagging type             = None" << std::endl;
        	break;
        case TREE_BAGGING_TYPE::SUBSAMPLE_WITH_REPLACEMENT:
        	o << "bagging type             = Subsample with replacement" << std::endl;
        	break;
        case TREE_BAGGING_TYPE::FIXED_RANDOM_SUBSET:
        	o << "bagging type 			   = Fixed subset (4 headpose, 30000 patches max per tree)" << std::endl;
        	break;
        default:
        	o << "bagging type             = UNDEFINED" << std::endl;
        	break;
        }
        o << "tree refinement          = " << hp.m_do_tree_refinement        << std::endl;
        o << "num random subsampling   = " << hp.m_num_random_samples_for_splitting << std::endl << std::endl;

//		switch (hp.m_adf_loss_classification)
//        {
//        case ADF_LOSS_CLASSIFICATION::GRAD_LOGIT:
//            o << "adf_loss_classification  = GradientBoost - Logit" << std::endl;
//            break;
//        case ADF_LOSS_CLASSIFICATION::GRAD_HINGE:
//            o << "adf_loss_classification  = GradientBoost - Hinge" << std::endl;
//            break;
//        case ADF_LOSS_CLASSIFICATION::GRAD_SAVAGE:
//            o << "adf_loss_classification  = GradientBoost - Savage" << std::endl;
//            break;
//        case ADF_LOSS_CLASSIFICATION::GRAD_EXP:
//            o << "adf_loss_classification  = GradientBoost - Exp" << std::endl;
//            break;
//        case ADF_LOSS_CLASSIFICATION::GRAD_TANGENT:
//            o << "adf_loss_classification  = GradientBoost - Tangent" << std::endl;
//            break;
//        case ADF_LOSS_CLASSIFICATION::ZERO_ONE:
//			o << "adf_loss_classification  = Zero-One" << std::endl;
//			break;
//        default:
//			o << "adf_loss_classification  = UNDEFINED" << std::endl;
//			break;
//        }

//		switch (hp.m_adf_loss_regression)
//		{
//        case ADF_LOSS_REGRESSION::SQUARED_LOSS:
//			o << "adf_loss_regression      = Squared Loss (regression)" << std::endl;
//			break;
//        case ADF_LOSS_REGRESSION::ABSOLUTE_LOSS:
//            o << "adf_loss_regression      = Absolute loss (regression)" << std::endl;
//            break;
//        case ADF_LOSS_REGRESSION::HUBER_LOSS:
//        	o << "adf_loss_regression      = Huber loss (regression" << std::endl;
//        	break;
//        default:
//			o << "adf_loss_regression      = UNDEFINED" << std::endl;
//			break;
//		}

//		switch (hp.m_adf_loss_mil)
//		{
//        case ADF_LOSS_MIL::NOISY_OR:
//            o << "adf_loss_mil             = Noisy OR (MIL)" << std::endl;
//            break;
//        case ADF_LOSS_MIL::NOISY_OR_GM:
//            o << "adf_loss_mil             = Noisy OR Geometric Mean (MIL)" << std::endl;
//            break;
//        case ADF_LOSS_MIL::NOISY_OR_PRIOR:
//            o << "adf_loss_mil             = Noisy OR Geometric Mean with prior (MIL)" << std::endl;
//            break;
//        default:
//            o << "adf_loss_mil             = UNDEFINED" << std::endl;
//            break;
//		}
//        o << "Huber-Loss delta         = " << hp.m_Huberloss_delta			 << std::endl;
//        o << "Shrinkage                = " << hp.m_shrinkage 				 << std::endl << std::endl;

        o << "===========================================================" << std::endl;

        return o;
    }


};



#endif /* RFCOREPARAMETERS_H_ */
