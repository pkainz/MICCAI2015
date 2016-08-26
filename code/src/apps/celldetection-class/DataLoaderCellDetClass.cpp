/*
 * DataLoaderCellDetClass.cpp
 *
 * Author: Philipp Kainz, Martin Urschler, Samuel Schulter, Paul Wohlhart, Vincent Lepetit
 * Institution: Medical University of Graz and Graz University of Technology, Austria
 *
 */

#include "DataLoaderCellDetClass.h"

// definition of the static member variable. important for accessing it wihtin a non-static method!!!
std::vector<std::vector<cv::Mat> > DataLoaderCellDetClass::image_feature_data;

DataLoaderCellDetClass::DataLoaderCellDetClass(AppContextCellDetClass* hp) : m_hp(hp) { }

DataLoaderCellDetClass::~DataLoaderCellDetClass() { }

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
 * - each pixel in the distance transform having a value of 255 (8bit) will be used as foreground in training
 * - from the set of zero-pixels in the distance transform, choose randomly a set of additional "background" locations
 *
 * @param trainImgFilenames a vector with image file names
 * @return the data set, consisting of single patch locations
 */
DataSet<SampleImgPatch, LabelMLClass> DataLoaderCellDetClass::LoadTrainData(std::vector<boost::filesystem::path>& trainImgFilenames)
{

    this->m_num_pos = 0;
    this->m_num_neg = 0;

    int n_mods_per_img = 4; // (NEW-Sami) ... how many modifiers per image (4 in this case with rotations, etc), suboptimal as it has to be pre-defined

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
         patch_locations.resize(trainImgFilenames.size()*n_mods_per_img); // (NEW-Sami)
    }

    // the train image file names are handed over by reference by the calling method
    if (!m_hp->quiet){
        std::cout << trainImgFilenames.size() << " training images available for extracting patches" << std::endl;
        std::cout << "feature list contains " << m_hp->image_feature_list.size() << " feature ids" << std::endl;
        std::cout << "Progress: " << std::flush;
    }

    // run through all training image names
    std::vector<DataSet<SampleImgPatch, LabelMLClass> > tmp_trainsets(trainImgFilenames.size()*n_mods_per_img);
#pragma omp parallel for
    for (size_t i = 0; i < trainImgFilenames.size(); i++)
    {
        string imgFilename = trainImgFilenames[i].filename().string();

        if (!m_hp->quiet){
            int progress = (int) round((double)i*100.0/(double)trainImgFilenames.size());
            std::cout << progress << "% ... " << imgFilename << std::flush;
        }

        std::vector<cv::Mat> src_image_vector, dt_image_vector;
        src_image_vector.clear();
        dt_image_vector.clear();

        // read source image
        cv::Mat src_img_raw = DataLoaderCellDetClass::ReadImageData(m_hp->path_traindata + "/" + imgFilename, false);
        // scaling
        cv::Mat src_img;
        cv::Size new_size = cv::Size((int)((double)src_img_raw.cols*m_hp->general_scaling_factor), (int)((double)src_img_raw.rows*m_hp->general_scaling_factor));
        resize(src_img_raw, src_img, new_size, 0, 0, cv::INTER_LINEAR);

        // extend the border if required
        ExtendBorder(m_hp, src_img, src_img, true);

        // read corresponding distance transform image (must have the same name)
        // reading just the name without extension: vector[i].stem()
        // load target image as grey
        cv::Mat dt_img_raw = DataLoaderCellDetClass::ReadImageData(this->m_hp->path_trainlabels + "/" + imgFilename, true);

        // scaling
        cv::Mat dt_img;
        resize(dt_img_raw, dt_img, new_size, 0, 0, cv::INTER_LINEAR);

        // extend the border if required
        ExtendBorder(m_hp, dt_img, dt_img, false);

        //        cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE );
        //        cv::imshow("Display window", dt_img);
        //        cv::waitKey(0);


        // add the src and dt image
        src_image_vector.push_back(src_img);
        dt_image_vector.push_back(dt_img);

        // EXTEND THE FOREGROUND PIXEL AMOUNT
        // take each image and flip it
        // vertically
        cv::Mat vertFlip_src, vertFlip_dt;
        cv::flip(src_img, vertFlip_src, 0);
        cv::flip(dt_img, vertFlip_dt, 0);
        // add the src and dt image
        src_image_vector.push_back(vertFlip_src);
        dt_image_vector.push_back(vertFlip_dt);

        // horizontally
        cv::Mat horzFlip_src, horzFlip_dt;
        cv::flip(src_img, horzFlip_src, 1);
        cv::flip(dt_img, horzFlip_dt, 1);
        // add the src and dt image
        src_image_vector.push_back(horzFlip_src);
        dt_image_vector.push_back(horzFlip_dt);

        // both directions
        cv::Mat bothFlip_src, bothFlip_dt;
        cv::flip(src_img, bothFlip_src, -1);
        cv::flip(dt_img, bothFlip_dt, -1);
        // add the src and dt image
        src_image_vector.push_back(bothFlip_src);
        dt_image_vector.push_back(bothFlip_dt);

        int src_vec_size = src_image_vector.size();

        // extract features for each src-dt pair (usually 4)
        for (size_t j = 0; j < src_vec_size; j++){
            // extract features for each source image
            std::vector<cv::Mat> img_features;
            DataLoaderCellDetClass::ExtractFeatureChannelsObjectDetection(src_image_vector[j], img_features, m_hp);

            // generate the patch positions
            if (!m_hp->load_dataset)
            {
                //patch_locations.push_back(this->GeneratePatchLocations(src_image_vector[j], dt_image_vector[j]));
                patch_locations[i*n_mods_per_img+j] = this->GeneratePatchLocations(src_image_vector[j], dt_image_vector[j]); // (NEW-Sami)
            }

            cout << "size patch_locations (at position)" << patch_locations.size() << " (" << (i*src_vec_size+j) << ")" << endl;

            // extract foreground image patches
            //this->ExtractPatches(m_trainset, src_image_vector[j], img_features,
            //                     patch_locations[(i*src_vec_size+j)], (i*src_vec_size+j), dt_image_vector[j]);
            this->ExtractPatches(tmp_trainsets[i*n_mods_per_img+j], src_image_vector[j], img_features,
                                 patch_locations[i*n_mods_per_img+j], (i*src_vec_size+j), dt_image_vector[j]);
        }

//		for (size_t j = 0; j < DataLoaderCellDetClass::image_feature_data.size(); j++)
//		{
//			cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE );
//			cv::imshow("Display window", DataLoaderCellDetClass::image_feature_data[j][0]);
//			cv::waitKey(0);
//		}
    }

    // (NEW-Sami) merge the data sets
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
    this->m_num_classes = 2; // set to 2

    // we just count the ORIGINAL channels, each original channel has an integral channel as well
    // thus, divide total feature data by 2!!
    //this->m_num_feature_channels = (int)DataLoaderCellDetClass::image_feature_data[0].size() / 2;
    this->m_num_feature_channels = (int)m_trainset[0]->m_sample.features.size() / 2; // (NEW-Sami)

    // 5) update the sample weights
    this->UpdateSampleWeights(m_trainset); // IMPORTANT FOR CLASSIFICATION TASKS!

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
 * @brief DataLoaderCellDetClass::GetTrainDataProperties
 * Get some properties of the training data, which are very important for the forest to
 * function correctly!
 *
 * @param num_samples
 * @param num_classes
 * @param num_feature_channels
 * @param num_fg total number of foreground samples
 * @param num_bg total number of background samples
 */
void DataLoaderCellDetClass::GetTrainDataProperties(int& num_samples, int& num_classes, int& num_feature_channels, int& num_fg, int& num_bg)
{
    num_samples = this->m_num_samples;
    num_classes = this->m_num_classes;
    num_feature_channels = this->m_num_feature_channels;
    num_fg = this->m_num_pos;
    num_bg = this->m_num_neg;
}

/**
 * Reads in the image from a file.
 *
 * @param imgpath the path to the file
 * @param loadgrayscale a flag, whether the target image should be loaded as greyscale
 * @return
 */
cv::Mat DataLoaderCellDetClass::ReadImageData(std::string imgpath, bool loadgrayscale)
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
void DataLoaderCellDetClass::ExtractFeatureChannelsObjectDetection(const cv::Mat& img, vector<cv::Mat>& vImg, AppContextCellDetClass* appcontext)
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

//        if (i < (int)vImg.size()/2){
//            cv::Mat tmp1;
//            vImg[i].convertTo(tmp1, CV_8U);
//            cv::imshow("Image Features", tmp1);
//        } else {
//            cv::Mat tmp;
//            double scale_fact = (double)vImg[i].at<float>(vImg[i].rows-1, vImg[i].cols-1);
//            cv::convertScaleAbs(vImg[i], tmp, 1.0/scale_fact);
//            cv::imshow("Image Features", tmp);
//        }

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
Eigen::MatrixXd DataLoaderCellDetClass::GeneratePatchLocations(const cv::Mat& src_img, const cv::Mat& dt_img)
{
    double ratio_additional_bg_patches = m_hp->ratio_additional_bg_patches;
    int patch_width = m_hp->patch_size[1];
    int patch_height = m_hp->patch_size[0];
    // compute offset from borders
    int offx = (int) (patch_width / 2);
    int offy = (int) (patch_height / 2);

    // store all black pixel locations for sampling
    vector<std::pair<int, int> > black_pixel_locations;
    // store all non-black pixel locations for sampling
    vector<std::pair<int, int> > non_black_pixel_locations;

    // run through the image, store top-left corner locations of zero and non-zero pixels at the center
    // filter according to the threshold_fg_bg
    for (unsigned int y = offy; y < dt_img.rows-offy; y++) {
        for (unsigned int x = offx; x < dt_img.cols-offx; x++) {
            if ((int)dt_img.at<uchar>(y,x) == 0){
                black_pixel_locations.push_back(std::make_pair(x-offx,y-offy));
            } else {
                // non-zero locations greater-equal than the defined threshold
                if ((int)dt_img.at<uchar>(y,x) == 255){
                    non_black_pixel_locations.push_back(std::make_pair(x-offx,y-offy));
                }
            }
        }
    }

    if (!m_hp->quiet){
        std::cout << "# non_black_pixel_locations: " << non_black_pixel_locations.size() << std::endl;
        std::cout << "# black_pixel_locations: " << black_pixel_locations.size() << std::endl;
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
    int num_additional_bg_patches_target = (int) (non_black_pixel_locations.size() * ratio_additional_bg_patches);

    std::vector<std::pair<int, int> > eligible_patches;
    eligible_patches.clear();

    int bpl = 0;
    while (num_additional_bg_patches_target > 0 // while we do not have enough patches
       && !(black_pixel_locations.empty()    // AND there are still any sampling location left
            )){

        std::pair<int, int> coords;

        // compute a random location and delete it from the black_pixel_locations vector.
        int idx = randInteger(0, black_pixel_locations.size()-1);
        coords = black_pixel_locations[idx];
        // erase the index from the vector
        black_pixel_locations.erase(black_pixel_locations.begin() + idx);
        bpl++;

        // if a proper location is found, add it to the vector
        eligible_patches.push_back(coords);

        num_additional_bg_patches_target--;
    }

    // allocate the patch location matrix (each row contains a top-left-corner (x,y) coordinate)
    Eigen::MatrixXd additional_patch_locations = Eigen::MatrixXd(eligible_patches.size(), 2); // rows, cols

    for (unsigned int i = 0; i < eligible_patches.size(); i++){

        std::pair<int, int> coords;
        coords = eligible_patches[i];

        additional_patch_locations(i, 0) = coords.first;
        additional_patch_locations(i, 1) = coords.second;
    }

    if (!m_hp->quiet){
        std::cout << "# background patches: ";
        std::cout << eligible_patches.size() << std::endl;
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

void DataLoaderCellDetClass::SavePatchPositions(std::string savepath, std::vector<Eigen::MatrixXd> patchpositions)
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

std::vector<Eigen::MatrixXd> DataLoaderCellDetClass::LoadPatchPositions(std::string loadpath)
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
Eigen::MatrixXi DataLoaderCellDetClass::LoadGTPositions(std::string loadpath)
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

void DataLoaderCellDetClass::SaveDetectionPositions(std::string savepath, Eigen::MatrixXi detected_locations)
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
void DataLoaderCellDetClass::ExtractPatches(DataSet<SampleImgPatch, LabelMLClass>& dataset, const cv::Mat& img,
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

        int target_class;
        // create the patch label information
        target_class = ((int)dt_img.at<uchar>(pt.y + offy, pt.x + offx) == 255)?1:0;

//        std::cout << target_class << std::endl;

//        cout << pt.x << ", " << pt.y << " --> ";

        int datastore_id = static_cast<int>(image_feature_data.size());
        SampleImgPatch patchdata;
        patchdata.features = img_features;
        patchdata.x = pt.x;
        patchdata.y = pt.y;
        patchdata.normalization_feature_mask = tmp_norm_mask;

        LabelMLClass patchlabel;
        patchlabel.class_label = target_class;
        patchlabel.gt_class_label = target_class;

        // finally add the sample to the data set
        LabelledSample<SampleImgPatch, LabelMLClass>* sample =
                new LabelledSample<SampleImgPatch, LabelMLClass>(patchdata, patchlabel, 1.0, datastore_id);
        dataset.AddLabelledSample(sample);

        // compute the balance of the dataset
        if (target_class == 1)
            this->m_num_pos++;
        else
            this->m_num_neg++;
    }
}

void DataLoaderCellDetClass::UpdateSampleWeights(DataSet<SampleImgPatch, LabelMLClass>& dataset)
{
    // get some statistics
    int num_pos = 0, num_neg = 0;
    for (size_t i = 0; i < dataset.size(); i++)
    {
        if (dataset[i]->m_label.class_label == 1)
            num_pos++;
        else
            num_neg++;
    }
    for (size_t i = 0; i < dataset.size(); i++)
    {
        if (dataset[i]->m_label.class_label == 1)
        {
            dataset[i]->m_label.class_weight = 1.0 / (double)num_pos / (double)m_num_classes * (double)(num_pos+num_neg);
            dataset[i]->m_label.class_weight_gt = dataset[i]->m_label.class_weight;
        }
        else
        {
            dataset[i]->m_label.class_weight = 1.0 / (double)num_neg / (double)m_num_classes * (double)(num_pos+num_neg);
            dataset[i]->m_label.class_weight_gt = dataset[i]->m_label.class_weight;
        }
    }
}
