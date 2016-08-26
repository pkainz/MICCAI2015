/*
 * DataLoaderCellDetRegr.cpp
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#include "DataLoaderCellDetRegr.h"

// definition of the static member variable. important for accessing it wihtin a non-static method!!!
std::vector<std::vector<cv::Mat> > DataLoaderCellDetRegr::image_feature_data;

DataLoaderCellDetRegr::DataLoaderCellDetRegr(AppContextCellDetRegr* hp) : m_hp(hp) { }

DataLoaderCellDetRegr::~DataLoaderCellDetRegr() { }

/**
 * Load the training data set from the file system.
 *
 * This method
 * - reads in all images from the 'path_traindata' directory,
 * - computes a series of features for each source image,
 * - reads in all corresponding distance transform images from 'path_trainlabels',
 * - extracts training patches, specified in a single txt file for each image in 'path_samplelocations'.
 *
 * Optionally, e.g. for the first time, the sample locations for each image are determined as follows:
 * - each non-zero pixel in the distance transform will be used for training
 * - from the set of zero-pixels in the distance transform, choose randomly a set of additional "background" locations
 *
 * @param trainImgFilenames a vector with image file names
 * @return the data set, consisting of single patch locations
 */
DataSet<SampleImgPatch, LabelMLRegr> DataLoaderCellDetRegr::LoadTrainData(std::vector<boost::filesystem::path>& trainImgFilenames)
{
    // 1) load pre-defined image data
    vector<MatrixXd> patch_locations;
    if (m_hp->load_dataset)
    {
        patch_locations = this->LoadPatchPositions(m_hp->path_fixedDataset);
        if (!m_hp->quiet)
            std::cout << "Loaded predefined random patch locations!" << std::endl;
    } else {
         if (!m_hp->quiet)
            std::cout << "Computing new patch locations from source images, this may take a while..." << std::endl;
         patch_locations.resize(trainImgFilenames.size()); // (NEW-Sami)
    }

    // the train image file names are handed over by reference by the calling method
    if (!m_hp->quiet){
        std::cout << trainImgFilenames.size() << " training images available for extracting patches" << std::endl;
        std::cout << "feature list contains " << m_hp->image_feature_list.size() << " feature ids" << std::endl;
        std::cout << "Progress: " << std::flush;
    }

    // run through all training image names
    std::vector<DataSet<SampleImgPatch, LabelMLRegr> > tmp_trainsets(trainImgFilenames.size()); // (NEW-Sami)
#pragma omp parallel for
    for (size_t i = 0; i < trainImgFilenames.size(); i++)
    {
        string imgFilename = trainImgFilenames[i].filename().string();

        if (!m_hp->quiet){
            int progress = (int) round((double)i*100.0/(double)trainImgFilenames.size());
            std::cout << progress << "% ... " << imgFilename << std::flush;
        }

        // read source image
        cv::Mat src_img_raw = DataLoaderCellDetRegr::ReadImageData(m_hp->path_traindata + "/" + imgFilename, false);
        // scaling
        cv::Mat src_img;
        cv::Size new_size = cv::Size((int)((double)src_img_raw.cols*m_hp->general_scaling_factor), (int)((double)src_img_raw.rows*m_hp->general_scaling_factor));
        resize(src_img_raw, src_img, new_size, 0, 0, cv::INTER_LINEAR);

        // extend the border if required
        ExtendBorder(m_hp, src_img, src_img, true);

//        cv::namedWindow("Display window", CV_WINDOW_NORMAL );
//        cv::imshow("Display window", src_img);
//        cv::waitKey(0);

        // extract features
        std::vector<cv::Mat> img_features;
        DataLoaderCellDetRegr::ExtractFeatureChannelsObjectDetection(src_img, img_features, m_hp);

        // read corresponding distance transform image (must have the same name)
        // reading just the name without extension: vector[i].stem()
        // load target image as grey
        cv::Mat dt_img_raw = DataLoaderCellDetRegr::ReadImageData(this->m_hp->path_trainlabels + "/" + imgFilename, true);

        // scaling
        cv::Mat dt_img;
        resize(dt_img_raw, dt_img, new_size, 0, 0, cv::INTER_LINEAR);

        // extend the border if required
        ExtendBorder(m_hp, dt_img, dt_img, false);

//        cv::namedWindow("Display window", CV_WINDOW_NORMAL );
//        cv::imshow("Display window", dt_img);
//        cv::waitKey(0);


        // generate the fg-patch positions
        if (!m_hp->load_dataset)
        {
            //patch_locations.push_back(this->GeneratePatchLocations(src_img, dt_img));
            patch_locations[i] = this->GeneratePatchLocations(src_img, dt_img); // (NEW-Sami)
        }

        // extract image patches (NEW-Sami)
        //this->ExtractPatches(m_trainset, src_img, img_features, patch_locations[i], i, dt_img);
        this->ExtractPatches(tmp_trainsets[i], src_img, img_features, patch_locations[i], i, dt_img);

//		for (size_t j = 0; j < DataLoaderCellDetRegr::image_feature_data.size(); j++)
//		{
//			cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE );
//			cv::imshow("Display window", DataLoaderCellDetRegr::image_feature_data[j][0]);
//			cv::waitKey(0);
//		}
    }

    // (NEW-Sami): merge the data sets by filling m_trainset
    for (size_t i=0; i < tmp_trainsets.size(); ++i)
    {
        for (size_t j=0; j < tmp_trainsets[i].size(); j++)
            m_trainset.AddLabelledSample(tmp_trainsets[i][j]);
    }

    if (!m_hp->quiet)
        std::cout << std::endl;

    if (!m_hp->quiet)
        std::cout << std::endl;

    // 4) store patch locations (for both pos and neg data!)
    if (m_hp->store_dataset)
    {
        this->SavePatchPositions(m_hp->path_fixedDataset, patch_locations);
        if (!m_hp->quiet)
            std::cout << "Stored random patch locations!" << std::endl;
    }

    this->m_num_samples = m_trainset.size();
    this->m_num_classes = 1; // set to 1, since we want to predict every pixel
    this->m_num_target_variables = 1; // total # of target variables is 1, distance transform values are univariat!
    // we just count the ORIGINAL channels, each original channel has an integral channel as well
    // thus, divide total feature data by 2!!
    //this->m_num_feature_channels = (int)DataLoaderCellDetRegr::image_feature_data[0].size() / 2;
    this->m_num_feature_channels = (int)m_trainset[0]->m_sample.features.size() / 2; // (NEW-Sami)

    // 5) update the sample weights
    //this->UpdateSampleWeights(m_trainset); // JUST IMPORTANT FOR CLASSIFICATION TASKS!

    {
        cout << "Clearing patch locations" << endl;
        patch_locations.clear();
        vector<MatrixXd> tmp;
        tmp.swap(patch_locations);
    }

    // 6) return filled dataset
    return m_trainset;
}


/**
 * @brief DataLoaderCellDetRegr::GetTrainDataProperties
 * Get some properties of the training data, which are very important for the forest to
 * function correctly!
 *
 * @param num_samples
 * @param num_classes
 * @param num_target_variables
 * @param num_feature_channels
 */
void DataLoaderCellDetRegr::GetTrainDataProperties(int& num_samples, int& num_classes,
                                                   int& num_target_variables, int& num_feature_channels)
{
    num_samples = this->m_num_samples;
    num_classes = this->m_num_classes;
    num_target_variables = this->m_num_target_variables;
    num_feature_channels = this->m_num_feature_channels;
}

/**
 * Reads in the image from a file.
 *
 * @param imgpath the path to the file
 * @param loadgrayscale a flag, whether the target image should be loaded as greyscale
 * @return
 */
cv::Mat DataLoaderCellDetRegr::ReadImageData(std::string imgpath, bool loadgrayscale)
{
    cv::Mat img;
    if (!loadgrayscale)
        img = cv::imread(imgpath);
    else
        img = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
        throw std::runtime_error("Error reading image: " + imgpath);
    return img;
    //	cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE );
    //	cv::imshow("Display window", img);
    //	cv::waitKey(0);
}

/**
 * Extract the feature channels for object detection.
 *
 * @param img the image
 * @param vImg the vector of feature channels of that image
 * @param appcontext the configuration
 */
void DataLoaderCellDetRegr::ExtractFeatureChannelsObjectDetection(const cv::Mat& img, vector<cv::Mat>& vImg, AppContextCellDetRegr* appcontext)
{
    bool use_integral_image = false;
    // check if random split function type list contains haar like features
    if (appcontext->use_random_split_function == 1){
        if (std::find(appcontext->split_function_type_list.begin(),
                      appcontext->split_function_type_list.end(),
                      SPLITFUNCTION_TYPE::HAAR_LIKE) != appcontext->split_function_type_list.end()){
                // only if haar-like features are used
                use_integral_image = true;
        }
    } else {
        // check the regular case (no random drawing)
        if (appcontext->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE) {
            // only if haar-like features are used
            use_integral_image = true;
        }
    }

// FC_GRAY     0
// FC_GABOR    1
// FC_SOBEL    2 // first derivatives
// FC_MIN_MAX  3
// FC_CANNY    4
// FC_NORM     5
// FC_LAB		6
// FC_GRAD2	7 // second derivatives
// FC_HOG		8
// FC_LUV      9
// FC_ORIENTED_GRAD_CHANFTRS 10
// FC_GRADMAGN 11
// FC_RGB		12
// FC_RELLOC   13 // relative location

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, CV_RGB2GRAY);

    FeatureGeneratorRGBImage fg;
    //fg.ExtractChannel(FC_RGB, use_integral_image, img, vImg);
    //fg.ExtractChannel(FC_LAB, use_integral_image, img, vImg);
    //fg.ExtractChannel(FC_LUV, use_integral_image, img, vImg);
    //fg.ExtractChannel(FC_SOBEL, use_integral_image, img_gray, vImg);
    //fg.ExtractChannel(FC_GRAD2, use_integral_image, img_gray, vImg);
    //fg.ExtractChannel(FC_GRADMAGN, use_integral_image, img_gray, vImg);
    //fg.ExtractChannel(FC_HOG, use_integral_image, img_gray, vImg);
    //fg.ExtractChannel(FC_NORM, use_integral_image, img_gray, vImg);
    //fg.ExtractChannel(FC_GABOR, use_integral_image, img_gray, vImg);

    // #######################################################################
    // compute the selected image features (no integral image)
    for (int i = 0; i < appcontext->image_feature_list.size(); i++){
        int feat_id = appcontext->image_feature_list[i];

        // some feature ids require color images
        if (feat_id == FC_RGB || feat_id == FC_LAB || feat_id == FC_LUV){
            fg.ExtractChannel(feat_id, false, img, vImg); // integral = false
        // others require greyscale images
        } else {
            fg.ExtractChannel(feat_id, false, img_gray, vImg); // integral = false
        }
    }

    // double the feature map, using a min and max filter and removing the
    if (appcontext->use_min_max_filters)
    {
        // number of feature channels before min-max filtering
        size_t num_channels_prior = vImg.size();

        // do the min max filtering of the channel (without integral image)
        for (size_t c = 0; c < num_channels_prior; c++)
            fg.ExtractChannel(FC_MIN_MAX, false, vImg[c], vImg); // integral = false

        // erase the unfiltered channels
        for (size_t c = 0; c < num_channels_prior; c++)
            vImg.erase(vImg.begin());
    }

    // set the number of feature channels
    appcontext->num_feature_channels = vImg.size();

    // #######################################################################
    // additionally compute the integral channels, if a function requires them
    if (use_integral_image){
        // compute the INTEGRAL image feature channels
        for (int i = 0; i < appcontext->image_feature_list.size(); i++){
            int feat_id = appcontext->image_feature_list[i];

            // some feature ids require color images
            if (feat_id == FC_RGB || feat_id == FC_LAB || feat_id == FC_LUV){
                fg.ExtractChannel(feat_id, true, img, vImg);
            // others require greyscale images
            } else {
                fg.ExtractChannel(feat_id, true, img_gray, vImg);
            }
        }
    }

    // apply a Gauss filter to all channels
    for (size_t i = 0; i < vImg.size(); i++)
    {
        cv::GaussianBlur(vImg[i], vImg[i], cv::Size(3, 3), 0, 0);
    }

    cout << endl;
    cout << "total number of feature channels (including integrals)=" << vImg.size() << endl;

//    cv::namedWindow("Image Features", CV_WINDOW_AUTOSIZE );
//    for (size_t i = 0; i < vImg.size(); i++)
//    {
//        for (int y = 0; y < 25; y++)
//        {
//            for (int x = 0; x < 25; x++)
//            {
//                cout << vImg[0].at<float>(y, x) << " ";
//            }
//            cout << endl;
//        }

//        cout << "Last entry: " << endl;
//        cout << vImg[0].at<float>(img.rows, img.cols) << endl;

//        cv::imshow("Image Features", vImg[i]);

        //cv::Mat tmp;
        //double scale_fact = (double)vImg[i].at<float>(vImg[i].rows-1, vImg[i].cols-1);
        //cv::convertScaleAbs(vImg[i], tmp, 1.0/scale_fact);
        //cv::imshow("Image Features", tmp);

//        cv::waitKey(0);
//    }
}





// PRIVATE / HELPER METHODS
/**
 * Samples a number of patch locations from the image and returns them.
 * The patch locations define the upper left corner of the patch.
 *
 * @param src_img the source image
 * @param dt_img the distance transform image
 * @return an Eigen::MatrixXd (nx2), holding top-left corner patch coordinates in each row
 */
Eigen::MatrixXd DataLoaderCellDetRegr::GeneratePatchLocations(const cv::Mat& src_img, const cv::Mat& dt_img)
{
    double ratio_additional_bg_patches = m_hp->ratio_additional_bg_patches;
    double threshold_fg_bg = m_hp->threshold_fg_bg;
    int patch_width = m_hp->patch_size[1];
    int patch_height = m_hp->patch_size[0];
    // compute offset from borders
    int offx = (int) (patch_width / 2);
    int offy = (int) (patch_height / 2);

    // store all black pixel locations for sampling
    vector<std::pair<int, int> > black_pixel_locations;
    // store all non-black pixel locations for sampling
    vector<std::pair<int, int> > non_black_pixel_locations;
    // store all near-by non-black pixel locations for sampling
    vector<std::pair<int, int> > nearby_non_black_pixel_locations;

    // run through the image, store top-left corner locations of zero and non-zero pixels at the center
    // filter according to the threshold_fg_bg
    for (unsigned int y = offy; y < dt_img.rows-offy; y++) {
        for (unsigned int x = offx; x < dt_img.cols-offx; x++) {
            if ((int)dt_img.at<uchar>(y,x) == 0){
                black_pixel_locations.push_back(std::make_pair(x-offx,y-offy));
            } else {
                // non-zero locations greater-equal than the defined threshold
                if ((int)dt_img.at<uchar>(y,x) >= (int) (threshold_fg_bg * 255)){
                    non_black_pixel_locations.push_back(std::make_pair(x-offx,y-offy));
                } else {
                    // all 'nearby' pixels
                    nearby_non_black_pixel_locations.push_back(std::make_pair(x-offx,y-offy));
                }
            }
        }
    }

    if (!m_hp->quiet){
        std::cout << "# non_black_pixel_locations: " << non_black_pixel_locations.size() << std::endl;
        std::cout << "# black_pixel_locations: " << black_pixel_locations.size() << std::endl;
        std::cout << "# nearby_center_locations: " << nearby_non_black_pixel_locations.size() << std::endl;
    }

    // allocate the patch location matrix for non-black pixels (each row contains a top-left-corner (x,y) coordinate)
    Eigen::MatrixXd non_black_patch_locations = Eigen::MatrixXd(non_black_pixel_locations.size(), 2); // rows, cols

    // take ALL filtered locations of non_black_pixel_locations and store it in the matrix
    for (unsigned int idx = 0; idx < non_black_pixel_locations.size(); idx++){
        std::pair<int, int> coords = non_black_pixel_locations[idx];
        non_black_patch_locations(idx, 0) = coords.first;
        non_black_patch_locations(idx, 1) = coords.second;
    }

    // if the ratio of additional background patches is set to < 0.0
    // sample a total of 50%
    if (ratio_additional_bg_patches < 0){
       ratio_additional_bg_patches = 0.5;
    }
    // otherwise take the ratio from the config

    // compute the number of additional patches to sample
    int num_additional_patches_target = (int) (non_black_pixel_locations.size() * ratio_additional_bg_patches);

    std::vector<std::pair<int, int> > eligible_patches;
    eligible_patches.clear();

    int nbnbpl = 0;
    int bpl = 0;
    // make a while loop and search for eligible patches
    while (num_additional_patches_target > 0 // while we do not have enough patches
       && !(black_pixel_locations.empty()    // AND there are still any sampling locations left
            && nearby_non_black_pixel_locations.empty())){

        std::pair<int, int> coords;

        // decide randomly with a 50% chance whether to draw from
        // 'pure' background or 'nearby' center locations
        if (randDouble() > 0.5){
            // check, if there are any black_pixel locations left for sampling
            if (black_pixel_locations.empty())
                continue;

            // compute a random index within the bounds of 'pure' background sample indices
            int idx = randInteger(0, black_pixel_locations.size()-1);
            coords = black_pixel_locations[idx];
            // erase the index from the vector
            black_pixel_locations.erase(black_pixel_locations.begin() + idx);
            bpl++;
        } else {
            // if the vector is empty stop searching
            if (nearby_non_black_pixel_locations.empty())
                continue;

            // compute a random index within the bounds of 'nearby' center sample indices
            int idx = randInteger(0, nearby_non_black_pixel_locations.size()-1);
            coords = nearby_non_black_pixel_locations[idx];
            // erase the index from the vector
            nearby_non_black_pixel_locations.erase(nearby_non_black_pixel_locations.begin() + idx);
            nbnbpl++;
        }

        // if a proper location is found, add it to the vector
        eligible_patches.push_back(coords);

        // reduce the number of required patches
        num_additional_patches_target--;
    }

    // allocate the patch location matrix (each row contains a top-left-corner (x,y) coordinate)
    Eigen::MatrixXd additional_patch_locations = Eigen::MatrixXd(eligible_patches.size(), 2); // rows, cols

    for (unsigned int i = 0; i < eligible_patches.size(); i++){
        std::pair<int, int> coords = eligible_patches[i];
        additional_patch_locations(i, 0) = coords.first;
        additional_patch_locations(i, 1) = coords.second;
    }

    if (!m_hp->quiet){
        std::cout << "# additional patches ('pure' background/'nearby center'): ";
        std::cout << eligible_patches.size() << " (" << bpl << "/" << nbnbpl << ")" << std::endl;
    }

    // concat the matrices vertically
    Eigen::MatrixXd patch_locations(non_black_patch_locations.rows()+additional_patch_locations.rows(), 2);
    patch_locations << non_black_patch_locations,
                       additional_patch_locations;

    {
        std::vector<std::pair<int, int> > tmp;
        eligible_patches.clear();
        tmp.swap(eligible_patches);
    }

    // return all patch locations in the image
    return patch_locations;
}

void DataLoaderCellDetRegr::SavePatchPositions(std::string savepath, std::vector<Eigen::MatrixXd> patchpositions)
{
    // write a new file with: #locations and locations (x,y)
    std::ofstream file;
    file.open(savepath.c_str(), ios::binary);
    file << patchpositions.size() << endl; // num images
    for (size_t i = 0; i < patchpositions.size(); i++)
    {
        file << patchpositions[i].rows() << " " << patchpositions[i].cols() << endl;
        file << patchpositions[i] << std::endl;
    }
    file.close();
}

std::vector<Eigen::MatrixXd> DataLoaderCellDetRegr::LoadPatchPositions(std::string loadpath)
{
    std::ifstream file;
    file.open(loadpath.c_str(), ios::in);
    unsigned int ni;
    file >> ni; // number of images
    vector<Eigen::MatrixXd> vec_patchpositions(ni);
    for (size_t i = 0; i < ni; i++)
    {
        int nr, nc; // number of rows and cols
        file >> nr >> nc;
        Eigen::MatrixXd patchpositions = Eigen::MatrixXd::Zero(nr, nc);
        for (int r = 0; r < nr; r++)
        {
            for (int c = 0; c < nc; c++)
            {
                int tmp;
                file >> tmp;
                patchpositions(r, c) = (double)tmp;
            }
        }
        vec_patchpositions[i] = patchpositions;
    }
    return vec_patchpositions;
}

// load the centers from a plain text file (serialized nx3 matrix)
// row->(x,y,label)
Eigen::MatrixXi DataLoaderCellDetRegr::LoadGTPositions(std::string loadpath)
{
    std::ifstream file;
    file.open(loadpath.c_str(), ios::in);

    int nr, nc, label; // number of rows and cols, label
    file >> nr >> nc >> label;
    // label should be 0!
    Eigen::MatrixXi patchpositions_withlabels = Eigen::MatrixXi::Zero(nr, nc);
    for (int r = 0; r < nr; r++)
    {
        for (int c = 0; c < nc; c++)
        {
            int tmp;
            file >> tmp;
            patchpositions_withlabels(r, c) = tmp;
        }
    }

    return patchpositions_withlabels;
}

void DataLoaderCellDetRegr::SaveDetectionPositions(std::string savepath, Eigen::MatrixXi detected_locations)
{
    // write a new file with: #detections, locations (x,y) and TP(1),FP(2),FN(3)
    std::ofstream file;
    file.open(savepath.c_str(), ios::binary);
    file << detected_locations.rows() << endl; // num elements go into first line

    file << detected_locations.rows() << " " << detected_locations.cols() << 0 << endl;
    file << detected_locations << std::endl; // write the matrix, end with a newline

    file.close(); // close the file
}


/**
 * Cut out the patches from the computed feature channels.
 *
 * @param dataset
 * @param img
 * @param img_features
 * @param patch_locations
 * @param img_index
 * @param dt_img the distance transform image to get the target variable from
 */
void DataLoaderCellDetRegr::ExtractPatches(DataSet<SampleImgPatch, LabelMLRegr>& dataset, const cv::Mat& img,
                                           const std::vector<cv::Mat>& img_features, const Eigen::MatrixXd& patch_locations,
                                           int img_index, const cv::Mat& dt_img)
{
    int patch_width = m_hp->patch_size[1];
    int patch_height = m_hp->patch_size[0];
    int offx = (int) (patch_width / 2);
    int offy = (int) (patch_height /2);

    int npatches = patch_locations.rows();

    if (m_hp->debug_on)
        std::cout << "nPatches: " << npatches << std::endl;


    cv::Mat temp_mask;
    temp_mask.create(cv::Size(img.rows, img.cols), CV_8U);
    temp_mask.setTo(cv::Scalar::all(1.0));
    cv::Mat tmp_norm_mask;
    cv::integral(temp_mask, tmp_norm_mask, CV_32F);

    for (int i = 0; i < npatches; i++)
    {
//        cout << i << " ";
        // patch position
        CvPoint pt = cvPoint(patch_locations(i, 0), patch_locations(i, 1));

//        cout << pt.x << ", " << pt.y << " --> ";

        int datastore_id = static_cast<int>(image_feature_data.size());
        SampleImgPatch patchdata;
        patchdata.features = img_features;
        patchdata.x = pt.x;
        patchdata.y = pt.y;
        patchdata.normalization_feature_mask = tmp_norm_mask;

         int target_label_value;
        // create the patch label information
        target_label_value = (int)dt_img.at<uchar>(pt.y + offy, pt.x + offx);

        LabelMLRegr patchlabel;
        Eigen::VectorXd _regr_target(1);
        _regr_target(0) = target_label_value;
        patchlabel.regr_target = _regr_target;
        patchlabel.regr_target_gt = _regr_target;
        patchlabel.regr_weight = 1.0;
        patchlabel.regr_weight_gt = 1.0;

        // finally add the sample to the data set
        LabelledSample<SampleImgPatch, LabelMLRegr>* sample = new LabelledSample<SampleImgPatch, LabelMLRegr>(patchdata, patchlabel, 1.0, datastore_id);
        dataset.AddLabelledSample(sample);
    }
}
