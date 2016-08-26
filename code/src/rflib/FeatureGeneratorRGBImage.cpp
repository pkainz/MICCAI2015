/*
 * FeatureGeneratorRGBImage.h
 *
 *  Created on: Sep 7, 2011
 *      Author: Matthias Dantone
 *
 */

#include "FeatureGeneratorRGBImage.h"


FeatureGeneratorRGBImage::FeatureGeneratorRGBImage() {}

void FeatureGeneratorRGBImage::ExtractChannel(int type, bool useIntegral, const cv::Mat& src, std::vector<cv::Mat>& channels)
{
	// ###########################################################
	// Feature: a simple gray scale image
	if (type == FC_GRAY)
	{
		if (useIntegral)
		{
			cv::Mat int_img;
			cv::integral(src, int_img, CV_32F);
			channels.push_back(int_img);
		}
		else
		{
			channels.push_back(src);
		}
	}
	// ###########################################################
	// Feature: histogram-equalized gray scale image
	else if (type == FC_NORM)
	{
		cv::Mat normal;
		cv::equalizeHist(src, normal);
		if (useIntegral)
		{
			cv::Mat int_img;
			cv::integral(normal, int_img, CV_32F);
			channels.push_back(int_img);
		}
		else
		{
			channels.push_back(normal);
		}

	}
	// ###########################################################
	// Feature: Gabor filters
	else if (type == FC_GABOR)
	{
		//check if kernels are initzialized
		if (reals.size() == 0)
			Init_gabor_kernels();

		int old_size = channels.size();
//		bool multithreaded = true;
//		if (multithreaded)
//		{
//			channels.resize(channels.size() + reals.size());
//			int num_treads = boost::thread::hardware_concurrency();
//			boost::thread_pool::executor e(num_treads);
//			for (unsigned int i = 0; i < reals.size(); i++)
//			{
//				e.submit(boost::bind(&FeatureChannelFactory::gabor_transform,
//						this, src, &channels[old_size + i], useIntegral, i, old_size));
//			}
//			e.join_all();
//		}
//		else
		{
			for (unsigned int i = 0; i < reals.size(); i++)
			{
				cv::Mat final;
				cv::Mat r_mat;
				cv::Mat i_mat;
				cv::filter2D(src, r_mat, CV_32F, reals[i]);
				cv::filter2D(src, i_mat, CV_32F, imags[i]);
				cv::pow(r_mat, 2, r_mat);
				cv::pow(i_mat, 2, i_mat);
				cv::add(i_mat, r_mat, final);
				cv::pow(final, 0.5, final);
				cv::normalize(final, final, 0, 1, CV_MINMAX, CV_32F);

				if (useIntegral)
				{
					cv::Mat img;
					final.convertTo(img, CV_8UC1, 255);

					cv::Mat integral_img;
					cv::integral(img, integral_img, CV_32F);
					channels.push_back(integral_img);
				}
				else
				{
					final.convertTo(final, CV_8UC1, 255);
					channels.push_back(final);
				}
			}
		}
	}

	// ###########################################################
	// Feature: Sobel ... simple gradients
	else if (type == FC_SOBEL)
	{
		cv::Mat sob_x_tmp;
		cv::Mat sob_y_tmp;

		cv::Sobel(src, sob_x_tmp, CV_16S, 1, 0);
		cv::Sobel(src, sob_y_tmp, CV_16S, 0, 1);
		cv::Mat sob_x(src.size(), CV_8U);
		cv::Mat sob_y(src.size(), CV_8U);
		cv::convertScaleAbs(sob_x_tmp, sob_x, 0.25);
		cv::convertScaleAbs(sob_y_tmp, sob_y, 0.25);

//		cv::Sobel(src, sob_x_tmp, CV_32F, 1, 0);
//		cv::Sobel(src, sob_y_tmp, CV_32F, 0, 1);
//		sob_x_tmp = (sob_x_tmp + 1020.0) / 2040.0;
//		//cv::normalize(sob_x_tmp, sob_x_tmp, 0, 1, CV_MINMAX, CV_32F);
//		sob_y_tmp = (sob_y_tmp + 1020.0) / 2040.0;
//		//cv::normalize(sob_y_tmp, sob_y_tmp, 0, 1, CV_MINMAX, CV_32F);
//		cv::Mat sob_x;
//		cv::Mat sob_y;
//		sob_x_tmp.convertTo(sob_x, CV_8U, 255);
//		sob_y_tmp.convertTo(sob_y, CV_8U, 255);

		if (useIntegral)
		{
			cv::Mat sob_x_int, sob_y_int;
			cv::integral(sob_x, sob_x_int, CV_32F);
			cv::integral(sob_y, sob_y_int, CV_32F);
			channels.push_back(sob_x_int);
			channels.push_back(sob_y_int);
		}
		else
		{
			channels.push_back(sob_x);
			channels.push_back(sob_y);
		}
	}

	// ###########################################################
	// Feature: Min-Max
	else if (type == FC_MIN_MAX)
	{
		cv::Mat kernel(cv::Size(3, 3), CV_8UC1);
		kernel.setTo(cv::Scalar(1));
		cv::Mat img_min(src.size(), CV_8U);
		cv::Mat img_max(src.size(), CV_8U);

		cv::erode(src, img_min, kernel);
		cv::dilate(src, img_max, kernel);

		if (useIntegral)
		{
			cv::Mat img_min_int, img_max_int;
			cv::integral(img_min, img_min_int, CV_32F);
			cv::integral(img_max, img_max_int, CV_32F);
			channels.push_back(img_min_int);
			channels.push_back(img_max_int);
		}
		else
		{
			channels.push_back(img_min);
			channels.push_back(img_max);
		}
	}

	// ###########################################################
	// Feature: Canny
	else if (type == FC_CANNY)
	{
		cv::Mat cannyImg;
		cv::Canny(src, cannyImg, -1, 5);
		if (useIntegral)
		{
			cv::Mat int_img;
			cv::integral(cannyImg, int_img, CV_32F);
			channels.push_back(int_img);
		}
		else
		{
			channels.push_back(cannyImg);
		}
	}

	// ###########################################################
	// Feature: LAB
	else if (type == FC_LAB)
	{
		cv::Mat img_lab;
		std::vector<cv::Mat> lab_channels(3);
		cv::cvtColor(src, img_lab, CV_RGB2Lab);
		cv::split(img_lab, lab_channels);
		for (size_t i = 0; i < 3; i++)
		{
			if (useIntegral)
			{
				cv::Mat imgInt;
				cv::integral(lab_channels[i], imgInt, CV_32F);
				channels.push_back(imgInt);
			}
			else
				channels.push_back(lab_channels[i]);
		}
	}

	// ###########################################################
	// Feature: LUV
	else if (type == FC_LUV)
	{
		cv::Mat img_luv;
		std::vector<cv::Mat> luv_channels(3);
		cv::cvtColor(src, img_luv, CV_RGB2Luv);
		cv::split(img_luv, luv_channels);
		for (size_t i = 0; i < 3; i++)
		{
			if (useIntegral)
			{
				cv::Mat imgInt;
				cv::integral(luv_channels[i], imgInt, CV_32F);
				channels.push_back(imgInt);
			}
			else
				channels.push_back(luv_channels[i]);
		}
	}

	// ###########################################################
	// Feature: RGB
	else if (type == FC_RGB)
	{
		cv::Mat img_rgb;
		std::vector<cv::Mat> rgb_channels(3);
		img_rgb = src;
		cv::split(img_rgb, rgb_channels);
		for (size_t i = 0; i < 3; i++)
		{
			if (useIntegral)
			{
				cv::Mat imgInt;
				cv::integral(rgb_channels[i], imgInt, CV_32F);
				channels.push_back(imgInt);
			}
			else
				channels.push_back(rgb_channels[i]);
		}
	}

	// ###########################################################
	// Feature: Second derivatives
	else if (type == FC_GRAD2)
	{
		cv::Mat sob_xx_tmp;
		cv::Mat sob_yy_tmp;

		cv::Sobel(src, sob_xx_tmp, CV_16S, 2, 0);
		cv::Sobel(src, sob_yy_tmp, CV_16S, 0, 2);
		cv::Mat sob_xx(src.size(), CV_8U);
		cv::Mat sob_yy(src.size(), CV_8U);
		cv::convertScaleAbs(sob_xx_tmp, sob_xx, 0.25);
		cv::convertScaleAbs(sob_yy_tmp, sob_yy, 0.25);

		//cv::Sobel(src, sob_xx_tmp, CV_32F, 2, 0);
		//cv::Sobel(src, sob_yy_tmp, CV_32F, 0, 2);
		////cv::normalize(sob_xx_tmp, sob_xx_tmp, 0, 1, CV_MINMAX, CV_32F);
		//sob_xx_tmp = (sob_xx_tmp + 1020.0) / 2040.0;
		////cv::normalize(sob_yy_tmp, sob_yy_tmp, 0, 1, CV_MINMAX, CV_32F);
		//sob_yy_tmp = (sob_yy_tmp + 1020.0) / 2040.0;
		//cv::Mat sob_xx;
		//cv::Mat sob_yy;
		//sob_xx_tmp.convertTo(sob_xx, CV_8U, 255);
		//sob_yy_tmp.convertTo(sob_yy, CV_8U, 255);

		if (useIntegral)
		{
			cv::Mat sob_xx_int, sob_yy_int;
			cv::integral(sob_xx, sob_xx_int, CV_32F);
			cv::integral(sob_yy, sob_yy_int, CV_32F);
			channels.push_back(sob_xx_int);
			channels.push_back(sob_yy_int);
		}
		else
		{
			channels.push_back(sob_xx);
			channels.push_back(sob_yy);
		}
	}

	// ###########################################################
	// Feature: HOG-like
	else if (type == FC_HOG)
	{
		cv::Mat sob_x;
		cv::Mat sob_y;
		cv::Sobel(src, sob_x, CV_16S, 1, 0, 3);
		cv::Sobel(src, sob_y, CV_16S, 0, 1, 3);

		int nbins = 9;
		int gauss_w = 5;
		std::vector<cv::Mat> hoglayers;
		this->CalculateHoGGall(sob_x, sob_y, hoglayers, nbins, gauss_w);

		if (useIntegral)
		{
			for (size_t i = 0; i < hoglayers.size(); i++)
			{
				cv::Mat int_img;
				cv::integral(hoglayers[i], int_img, CV_32F);
				channels.push_back(int_img);
			}
		}
		else
		{
			for (size_t i = 0; i < hoglayers.size(); i++)
				channels.push_back(hoglayers[i]);
		}
	}

	// ###########################################################
	// Feature: Binned Oriented Gradients - Channel Features (Dollar, Benenson, ...)
	else if (type == FC_ORIENTED_GRAD_CHANFTRS)
	{
		cv::Mat sob_x;
		cv::Mat sob_y;
		cv::Sobel(src, sob_x, CV_32F, 1, 0, 3);
		cv::Sobel(src, sob_y, CV_32F, 0, 1, 3);
		// min/max values: 4*255 -> 1020

		cv::Mat magnitude;
		cv::magnitude(sob_x, sob_y, magnitude);
		// min/max values: 0/1020*sqrt(2)=1442,5

		cv::Mat angle;
		cv::phase(sob_x, sob_y, angle, true);
		// min/max values: 0/360

		//double minVal, maxVal;
		//cv::minMaxIdx(angle, &minVal, &maxVal);
		//std::cout << minVal << ", " << maxVal << std::endl;

		int nbins = 6;
		std::vector<float> bin_centers(nbins);
		bin_centers[0] = 15.0;
		bin_centers[1] = 45.0;
		bin_centers[2] = 75.0;
		bin_centers[3] = 105.0;
		bin_centers[4] = 135.0;
		bin_centers[5] = 165.0;
		double bin_size = 30.0;
		float diff, influence;
		std::vector<cv::Mat> gradlayers_tmp(nbins);
		for (size_t i = 0; i < gradlayers_tmp.size(); i++)
		{
			gradlayers_tmp[i] = cv::Mat(src.size(), CV_32F);
			gradlayers_tmp[i].setTo(cv::Scalar::all(0.0));
		}
		for (int x = 0; x < magnitude.cols; x++)
		{
			for (int y = 0; y < magnitude.rows; y++)
			{
				// map all orientation above 180°
				float o = angle.at<float>(y, x);
				if (o >= 180.0)
					o -= 180.0;

				//std::cout << x << ", " << y << ": o=" << o << ", m=" << magnitude.at<float>(y, x) << ": " << std::endl;

				// bin the orientation
				for (size_t i = 0; i < bin_centers.size(); i++)
				{
					diff = std::abs(o - bin_centers[i]);

					//std::cout << " - " << diff;

					if (diff < bin_size)
					{
						influence = 1.0 - diff/bin_size;
						gradlayers_tmp[i].at<float>(y, x) = magnitude.at<float>(y, x) * influence;
						//std::cout << " -> in with inf=" << influence << " -> val=" << gradlayers_tmp[i].at<float>(y, x);
					}
					//std::cout << std::endl;
				}

				// check out the special cases o=[0:15], o=[165:180]
				if (o < bin_centers[0])
				{
					// diff to last entry! circular
					diff = bin_size - std::abs(o - bin_centers[0]);
					influence = 1.0 - diff/bin_size;
					gradlayers_tmp[nbins-1].at<float>(y, x) = magnitude.at<float>(y, x) * influence;
				}

				if (o > bin_centers[nbins-1])
				{
					// diff to first entry! circular
					diff = bin_size - std::abs(o - bin_centers[nbins-1]);
					influence = 1.0 - diff/bin_size;
					gradlayers_tmp[0].at<float>(y, x) = magnitude.at<float>(y, x) * influence;
				}

			}
		}

		// min/max values for gradlayers_tmp:
		// 0 / 1442.5
		// -> scaling factor: 0.176
		// However, we don't use the full range! -> other scaling factor: 0.90, the rest is clipped!
		// empirical estimated from a few images
		std::vector<cv::Mat> gradlayers(nbins);
		for (size_t i = 0; i < gradlayers.size(); i++)
		{
			cv::convertScaleAbs(gradlayers_tmp[i], gradlayers[i], 0.90);
		}

		if (useIntegral)
		{
			for (size_t i = 0; i < gradlayers.size(); i++)
			{
				cv::Mat int_img;
				cv::integral(gradlayers[i], int_img, CV_32F);
				channels.push_back(int_img);
			}
		}
		else
		{
			for (size_t i = 0; i < gradlayers.size(); i++)
				channels.push_back(gradlayers[i]);
		}
	}

	// ###########################################################
	// Feature: Gradient Magnitude
	else if (type == FC_GRADMAGN)
	{
		cv::Mat sob_x;
		cv::Mat sob_y;
		cv::Sobel(src, sob_x, CV_32F, 1, 0, 3);
		cv::Sobel(src, sob_y, CV_32F, 0, 1, 3);
		// 255 * 4
		// theory min/max values: -1020/+1020
		// double minVal = -1020.0, maxVal = +1020.0;

		cv::Mat magnitude_tmp;
		cv::magnitude(sob_x, sob_y, magnitude_tmp);
		// new max value: (+/-)1020 * sqrt(2) = 1442,5;
		// new min value: 0

		// new scaling factor for conversion:
		// 255 / 1443 = 0.1767 -> 0.176
		// However, we don't use the full range! -> other scaling factor: 0.90, the rest is clipped!
		// empirical estimated from a few images
		cv::Mat magnitude(src.size(), CV_8U);
		cv::convertScaleAbs(magnitude_tmp, magnitude, 0.90);


		//cv::normalize(magnitude_tmp, magnitude_tmp, 0, 1, CV_MINMAX, CV_32F);
		//cv::Mat magnitude;
		//magnitude_tmp.convertTo(magnitude, CV_8U, 255);


		if (useIntegral)
		{
			cv::Mat int_img;
			cv::integral(magnitude, int_img, CV_32F);
			channels.push_back(int_img);
		}
		else
		{
			channels.push_back(magnitude);
		}
	}

	// ###########################################################
	// Feature: relative location feature
	else if (type == FC_RELLOC)
	{
		// src is given ...
		cv::Mat relloc_x(src.size(), CV_8U);
		cv::Mat relloc_y(src.size(), CV_8U);
		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				relloc_x.at<uchar>(y, x) = (uchar)(int)((double)x / (double)src.cols * 255.0);
				relloc_y.at<uchar>(y, x) = (uchar)(int)((double)y / (double)src.rows * 255.0);
			}
		}

		if (useIntegral)
		{
			cv::Mat int_img_x;
			cv::Mat int_img_y;
			cv::integral(relloc_x, int_img_x, CV_32F);
			cv::integral(relloc_y, int_img_y, CV_32F);
			channels.push_back(int_img_x);
			channels.push_back(int_img_y);
		}
		else
		{
			channels.push_back(relloc_x);
			channels.push_back(relloc_y);
		}
	}

	else
	{
		throw std::runtime_error("FeatureGenerator: Unknown feature channel");
	}
}




// Helper methods for Gabor features
void FeatureGeneratorRGBImage::Gabor_transform(const cv::Mat& src, cv::Mat* dst, bool useIntegral, int index, int old_size) const
{
	cv::Mat final;
	cv::Mat r_mat;
	cv::Mat i_mat;
    cv::filter2D(src, r_mat, CV_32S, reals[index]);
    cv::filter2D(src, i_mat, CV_32S, imags[index]);
    cv::pow(r_mat, 2, r_mat);
    cv::pow(i_mat, 2, i_mat);
    cv::add(i_mat, r_mat, final);
    cv::pow(final, 0.5, final);
    cv::normalize(final, final, 0, 1, CV_MINMAX);

    if (useIntegral)
    {
		cv::Mat img;
		final.convertTo(img, CV_8UC1, 255);
		cv::Mat integral_img;
		cv::integral(img, integral_img, CV_32F);
		*dst = integral_img;
    }
    else
    {
		final.convertTo(final, CV_8UC1, 255);
		*dst = final;
    }
}

void FeatureGeneratorRGBImage::Init_gabor_kernels()
{
    //create kernels
    int NuMin = 0;
    int NuMax = 4;
    int MuMin = 0;
    int MuMax = 8;
    double sigma = 1. / 2.0 * CV_PI;
    double dF = sqrt(2.0);

    int iMu = 0;
    int iNu = 0;

    for (iNu = NuMin; iNu <= NuMax; iNu+=1)
    	for (iMu = MuMin; iMu <= MuMax; iMu+=2)
			CreateKernel(iMu, iNu, sigma, dF);
}

void FeatureGeneratorRGBImage::CreateKernel(int iMu, int iNu, double sigma, double dF)
{
    //Initilise the parameters
    double F = dF;
    double k = (CV_PI / 2) / pow(F, (double) iNu);
    double phi = CV_PI * iMu / 8;

    double width = round((sigma / k) * 6 + 1);
    if (fmod(width, 2.0) == 0.0)
    	width++;

    //create kernel
    cv::Mat m_real = cv::Mat(width, width, CV_32FC1);
    cv::Mat m_imag = cv::Mat(width, width, CV_32FC1);

    int x, y;
    double dReal;
    double dImag;
    double dTemp1, dTemp2, dTemp3;

    int off_set = (width - 1) / 2;
    for (int i = 0; i < width; i++)
    {
		for (int j = 0; j < width; j++)
		{
			x = i - off_set;
			y = j - off_set;
			dTemp1 = (pow(k, 2) / pow(sigma, 2)) * exp(-(pow((double) x, 2) + pow((double) y, 2)) * pow(k, 2) / (2 * pow(sigma, 2)));
			dTemp2 = cos(k * cos(phi) * x + k * sin(phi) * y) - exp(-(pow(sigma, 2) / 2));
			dTemp3 = sin(k * cos(phi) * x + k * sin(phi) * y);
			dReal = dTemp1 * dTemp2;
			dImag = dTemp1 * dTemp3;
			m_real.at<float>(j, i) = dReal;
			m_imag.at<float>(j, i) = dImag;
		}
    }
    reals.push_back(m_real);
    imags.push_back(m_imag);
};




// Helper methods for HoG features
void FeatureGeneratorRGBImage::CalculateHoGGall(cv::Mat sob_x, cv::Mat sob_y, std::vector<cv::Mat>& hoglayers, int nbins, int gauss_w)
{
	// 1) calcualte magnitude and orientation
	cv::Mat magnitude(sob_x.size(), CV_8U);
	cv::Mat orientation(sob_x.size(), CV_8U);
	for (int y = 0; y < magnitude.rows; y++)
	{
		for (int x = 0; x < magnitude.cols; x++)
		{
			// calculate magnitude
			float dx = (float)sob_x.at<short>(y, x);
			dx = dx + (float)copysign(0.000001f, dx); // avoid division by zero
			float dy = (float)sob_y.at<short>(y, x);
			float c_magn = sqrt(dx*dx + dy*dy);
			//if (c_magn > 255)
			//	c_magn = 255;
			magnitude.at<uchar>(y, x) = (uchar)c_magn;
			// calculate orientation
			float c_orie = (atan(dy/dx) + 3.14159265f/2.0f) * 80;
			orientation.at<uchar>(y, x) = (uchar)c_orie;
		}
	}
//	cv::namedWindow("Image Features", CV_WINDOW_AUTOSIZE );
//	cv::imshow("Image Features", magnitude);
//	cv::waitKey(0);
//	cv::imshow("Image Features", orientation);
//	cv::waitKey(0);



	// 12) create the Gauss filter
	cv::Mat gauss_filter(gauss_w, gauss_w, CV_32F);
	float a = -(gauss_w-1)/2.0;
	float sigma2 = 2*(0.5*gauss_w)*(0.5*gauss_w);
	float count = 0;
	for (int x = 0; x < gauss_w; x++)
	{
		for (int y = 0; y < gauss_w; y++)
		{
			float tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
			count += tmp;
			gauss_filter.at<float>(y, x) = tmp;
		}
	}
	gauss_filter *= 1.0 / count;

	// 2) iterate all pixels and find the corresponding bin contributions
	float binsize = (3.14159265f*80.0f)/(float)nbins;
	hoglayers.resize(nbins);
	for (size_t l = 0; l < hoglayers.size(); l++)
	{
		hoglayers[l] = cv::Mat(sob_x.size(), CV_8U);
		hoglayers[l].setTo(cv::Scalar::all(0.0));
	}
	int gauss_w_off = int(gauss_w / 2);
	for (int y = 0; y < (magnitude.rows-gauss_w); y++)
	{
		for (int x = 0; x < (magnitude.cols-gauss_w); x++)
		{
			// start the gauss filter loops
			std::vector<double> bin_values(nbins, 0.0);
			for (int gy = 0; gy < gauss_w; gy++)
			{
				for (int gx = 0; gx < gauss_w; gx++)
				{
					float m = (float)magnitude.at<uchar>(y+gy, x+gx) * gauss_filter.at<float>(gy, gx);
					float o = (float)orientation.at<uchar>(y+gy, x+gx) / binsize;
					int bin1 = (int)o;
					int bin2;
					float delta = o - bin1 - 0.5;
					if (delta < 0)
					{
						bin2 = bin1 < 1 ? nbins-1 : bin1-1;
						delta = -delta;
					}
					else
					{
						bin2 = bin1 < nbins-1 ? bin1+1 : 0;
					}
					//if (bin1 == 1 || bin2 == 2)
					//	std::cout << "juhu" << std::endl;
					//std::cout << bin1 << ", " << bin2 << ", " << m << ", " << o << ", " << delta << std::endl;
					//std::cout << (uchar)((1.0-delta)*m) << std::endl;
					//std::cout << (uchar)(delta*m) << std::endl;
					bin_values[bin1] += (1.0-delta)*m;
					bin_values[bin2] += delta*m;
				}
			}
			for (size_t b = 0; b < bin_values.size(); b++)
				hoglayers[b].at<uchar>(y+gauss_w_off, x+gauss_w_off) = (uchar)bin_values[b];
		}
	}
}








