/*
 * SplitFunctionImgPatch.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITFUNCTIONIMGPATCH_CPP_
#define SPLITFUNCTIONIMGPATCH_CPP_


#include "SplitFunctionImgPatch.h"




template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::SplitFunctionImgPatch(AppContext* appcontextin) : m_appcontext(appcontextin)
{
}

template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::~SplitFunctionImgPatch()
{
}



template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
void SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::SetRandomValues()
{
	// some initializations
	int min_w, min_h, max_w, max_h;
	int min_x, min_y, max_x, max_y;

    // if set, draw a random split function from the list
    // and set the member of SplitFunctionImgPatch
    if (m_appcontext->use_random_split_function){
        int num_available_split_functions = m_appcontext->split_function_type_list.size();
        this->m_splitfunction_type = m_appcontext->split_function_type_list[randInteger(0, num_available_split_functions-1)];
    } else {
        // use the default implementation
        this->m_splitfunction_type = m_appcontext->split_function_type;
    }

	// get some patch sub-regions
    switch (this->m_splitfunction_type){
	case SPLITFUNCTION_TYPE::SINGLEPIXELTEST:
		this->px1.x = randInteger(0, m_appcontext->patch_size[1]-1);
		this->px1.y = randInteger(0, m_appcontext->patch_size[0]-1);
		break;
	case SPLITFUNCTION_TYPE::PIXELPAIRTEST:
		this->px1.x = randInteger(0, m_appcontext->patch_size[1]-1);
		this->px1.y = randInteger(0, m_appcontext->patch_size[0]-1);
		this->px2.x = randInteger(0, m_appcontext->patch_size[1]-1);
		this->px2.y = randInteger(0, m_appcontext->patch_size[0]-1);
		break;
	case SPLITFUNCTION_TYPE::PIXELPAIRTESTCONDITIONED:
		this->px1.x = randInteger(0, m_appcontext->patch_size[1]-1);
		this->px1.y = randInteger(0, m_appcontext->patch_size[0]-1);
		int min_x, max_x, min_y, max_y;
		max_w = 10;
		min_x = max(0, px1.x - max_w);
		max_x = min(m_appcontext->patch_size[1]-1, px1.x + max_w);
		min_y = max(0, px1.y - max_w);
		max_y = min(m_appcontext->patch_size[0]-1, px1.y + max_w);
		this->px2.x = randInteger(min_x, max_x);
		this->px2.y = randInteger(min_y, max_y);
		break;
	case SPLITFUNCTION_TYPE::HAAR_LIKE:
		min_w = 1;
		min_h = 1;
		//max_w = static_cast<int>((double)m_appcontext->patch_size[1]*0.5);
		//max_h = static_cast<int>((double)m_appcontext->patch_size[0]*0.5);
		max_w = static_cast<int>((double)m_appcontext->patch_size[1]*0.90);
		max_h = static_cast<int>((double)m_appcontext->patch_size[0]*0.90);
		this->re1.width = randInteger(min_w, max_w);
		this->re1.height = randInteger(min_w, max_w);
		this->re1.x = randInteger(0, m_appcontext->patch_size[1]-re1.width);
		this->re1.y = randInteger(0, m_appcontext->patch_size[0]-re1.height);
		this->re2.width = randInteger(min_w, max_w);
		this->re2.height = randInteger(min_w, max_w);
		this->re2.x = randInteger(0, m_appcontext->patch_size[1]-re2.width);
		this->re2.y = randInteger(0, m_appcontext->patch_size[0]-re2.height);
		break;
	case SPLITFUNCTION_TYPE::ORDINAL:
//		this->pxs.resize(m_appcontext->ordinal_split_k);
//		for (size_t k = 0; k < pxs.size(); k++)
//		{
//			pxs[k].x = randInteger(0, m_appcontext->patch_size[1]-1);
//			pxs[k].y = randInteger(0, m_appcontext->patch_size[0]-1);
//		}
        // ORDINAL IS IMPLEMENTED HERE, BUT NOT IN GETRESPONSE?????
        throw std::logic_error("SplitFunction (set random value): Ordinal split functions not implemented yet!");
		break;
	default:
		throw std::runtime_error("SplitFunction: unknown split-type not implemented");
	}

	// define the feature channels
	this->ch1 = randInteger(0, m_appcontext->num_feature_channels-1);
	if (m_appcontext->split_channel_selection == 0)
		// use the same channel
		this->ch2 = this->ch1;
	else if (m_appcontext->split_channel_selection == 1)
		// use 2 random channels
		this->ch2 = randInteger(0, m_appcontext->num_feature_channels-1);
	else if (m_appcontext->split_channel_selection == 2)
	{
		// 50/50 chance of using same channel or using 2 random channels.
		if (randDouble() > 0.5)
			this->ch2 = this->ch1;
		else
			this->ch2 = randInteger(0, m_appcontext->num_feature_channels-1);
	}
	else
		throw std::runtime_error("SplitFunction: unknown channel-selection type");
}


template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
void SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::SetThreshold(double inth)
{
	this->m_th = inth;
}


template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
void SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::SetSplit(SplitFunctionImgPatch* spfin)
{
	// copy the split!
	ch1 = spfin->ch1;
	ch2 = spfin->ch2;
	px1 = spfin->px1;
	px2 = spfin->px2;
	re1 = spfin->re1;
	re2 = spfin->re2;
	pxs.resize(spfin->pxs.size());
	for (size_t i = 0; i < pxs.size(); i++)
		pxs[i] = spfin->pxs[i];
	m_th = spfin->m_th;

    // set the split function type
    m_splitfunction_type = spfin->m_splitfunction_type;
}


template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
int SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::Split(SampleImgPatch& sample)
{
	if (this->GetResponse(sample) < this->m_th)
		return 0;
	else
		return 1;
}


template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
double SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::CalculateResponse(SampleImgPatch& sample)
{
	return this->GetResponse(sample);
}

template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
void SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::Print(){

    cout << ch1 << " " << ch2 << " ";
    cout << px1.x << " " << px1.y << " ";
    cout << px2.x << " " << px2.y << " ";
    cout << re1.x << " " << re1.y << " " << re1.width << " " << re1.height << " ";
    cout << re2.x << " " << re2.y << " " << re2.width << " " << re2.height << " ";
    cout << pxs.size() << " ";
    for (size_t i = 0; i < pxs.size(); i++)
        cout << pxs[i].x << " " << pxs[i].y << " ";
    cout << m_th << " ";
    // write out the split function type
    cout << m_splitfunction_type;
}

template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
void SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::Save(std::ofstream& out)
{
	out << ch1 << " " << ch2 << " ";
	out << px1.x << " " << px1.y << " ";
	out << px2.x << " " << px2.y << " ";
	out << re1.x << " " << re1.y << " " << re1.width << " " << re1.height << " ";
	out << re2.x << " " << re2.y << " " << re2.width << " " << re2.height << " ";
	out << pxs.size() << " ";
	for (size_t i = 0; i < pxs.size(); i++)
		out << pxs[i].x << " " << pxs[i].y << " ";
    out << m_th << " ";
    // write out the split function type
    out << m_splitfunction_type << " ";
    out << endl;
}


template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
void SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::Load(std::ifstream& in)
{
    in >> ch1 >> ch2;
	in >> px1.x >> px1.y;
	in >> px2.x >> px2.y;
	in >> re1.x >> re1.y >> re1.width >> re1.height;
	in >> re2.x >> re2.y >> re2.width >> re2.height;
	int npxs;
	in >> npxs;
	pxs.resize(npxs);
	for (size_t i = 0; i < pxs.size(); i++)
		in >> pxs[i].x >> pxs[i].y;
    in >> m_th;
    // read in split function type
    in >> m_splitfunction_type;
}


// Private / Helper methods
template<typename BaseType, typename BaseTypeIntegral, typename AppContext>
double SplitFunctionImgPatch<BaseType, BaseTypeIntegral, AppContext>::GetResponse(SampleImgPatch& sample)
{
	// get the datastore-relevant information from the sample
	int patch_x = sample.x;
	int patch_y = sample.y;

	// calculate the feature response
	BaseType val1, val2;
	BaseTypeIntegral val1int, val2int;
	//int area1int, area2int;
    // integral image base types (float or double)
	BaseTypeIntegral area1int, area2int;
    int channel_offset, ch1_int, ch2_int;
    switch (this->m_splitfunction_type)
	{
	case SPLITFUNCTION_TYPE::SINGLEPIXELTEST:
		val1 = sample.features[ch1].at<BaseType>(patch_y+px1.y, patch_x+px1.x);
		return (double)val1;
		break;
	case SPLITFUNCTION_TYPE::PIXELPAIRTEST:
	case SPLITFUNCTION_TYPE::PIXELPAIRTESTCONDITIONED:
		val1 = sample.features[ch1].at<BaseType>(patch_y+px1.y, patch_x+px1.x);
		//std::cout << "Accessing: " << patch_y << "+" << px1.y << ", " << patch_x << "+" << px1.x << " | featmap: " << sample.features[ch1].rows << " x " << sample.features[ch1].cols << " | -->" << (int)val1 << std::endl;
		val2 = sample.features[ch2].at<BaseType>(patch_y+px2.y, patch_x+px2.x);
		return ((double)val1 - (double)val2);
		break;
	case SPLITFUNCTION_TYPE::HAAR_LIKE:
        // use an offset for the integral feature channels
        channel_offset = m_appcontext->num_feature_channels;
        //cout << "split can choose from " << channel_offset << " feature channels " << endl;
        ch1_int = ch1 + channel_offset;
        ch2_int = ch2 + channel_offset;

        // AT THIS POINT, WE REQUIRE INTEGRAL IMAGES, so we take the integral image positions
        // from the feature map (with the offset)
        val1int = sample.features[ch1_int].at<BaseTypeIntegral>(patch_y+re1.y, patch_x+re1.x) + sample.features[ch1_int].at<BaseTypeIntegral>(patch_y+re1.y+re1.height-1, patch_x+re1.x+re1.width-1) - sample.features[ch1_int].at<BaseTypeIntegral>(patch_y+re1.y+re1.height-1, patch_x+re1.x) - sample.features[ch1_int].at<BaseTypeIntegral>(patch_y+re1.y, patch_x+re1.x+re1.width-1);
        // caution: the normalization-feature-mask always starts at 0,0 and has the size of the patch!
		area1int = (sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re1.y, patch_x+re1.x) + sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re1.y+re1.height-1, patch_x+re1.x+re1.width-1) - sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re1.y+re1.height-1, patch_x+re1.x) - sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re1.y, patch_x+re1.x+re1.width-1));
		area1int = max((BaseTypeIntegral)1.0, area1int);
        val2int = sample.features[ch2_int].at<BaseTypeIntegral>(patch_y+re2.y, patch_x+re2.x) + sample.features[ch2_int].at<BaseTypeIntegral>(patch_y+re2.y+re2.height-1, patch_x+re2.x+re2.width-1) - sample.features[ch2_int].at<BaseTypeIntegral>(patch_y+re2.y+re2.height-1, patch_x+re2.x) - sample.features[ch2_int].at<BaseTypeIntegral>(patch_y+re2.y, patch_x+re2.x+re2.width-1);
		// caution: the normalization-feature-mask always starts at 0,0 and has the size of the patch!
		area2int = (sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re2.y, patch_x+re2.x) + sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re2.y+re2.height-1, patch_x+re2.x+re2.width-1) - sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re2.y+re2.height-1, patch_x+re2.x) - sample.normalization_feature_mask.at<BaseTypeIntegral>(patch_y+re2.y, patch_x+re2.x+re2.width-1));
		area2int = max((BaseTypeIntegral)1.0, area2int);
		return ((double)val1int / (double)area1int - (double)val2int / (double)area2int);
		break;
	case SPLITFUNCTION_TYPE::ORDINAL:
		throw std::logic_error("SplitFunction (getresponse): Ordinal split functions not implemented yet!");
		return 0.0;
		break;
	default:
        cout << m_splitfunction_type << " --> ";
		throw std::runtime_error("SplitFunction (getresponse): unknown split-type not implemented");
		return 0.0;
	}
}

#endif /* SPLITFUNCTINIMGPATCH */

