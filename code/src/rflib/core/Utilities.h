/*
 * Utilities.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <stdexcept>
#include <string>
#include <math.h>
#include <time.h>
#include <fstream>
#include <sys/time.h>
#include <random/uniform.h>
#include <random/normal.h>
#include <eigen3/Eigen/Core>
#include <algorithm>
#include <set>
#include <unistd.h> // required by Mac OSX??

using namespace std;
using namespace ranlib;
using namespace Eigen;



inline unsigned int getDevRandom()
{
    ifstream devFile("/dev/urandom", ios::binary);
    unsigned int outInt = 0;
    char tempChar[sizeof(outInt)];

    devFile.read(tempChar, sizeof(outInt));
    outInt = atoi(tempChar);

    devFile.close();

    return outInt;
}


// TODO: use a platform independent random number generator!
inline double randDouble()
{
    static int seedCount = 0;
	static Uniform<double> uniformRandGen;
	if (!seedCount)
	{
		struct timeval TV;
		unsigned int curTime;
		gettimeofday(&TV, NULL);
		curTime = (unsigned int) TV.tv_usec;
		uniformRandGen.seed((unsigned int) time(NULL) + curTime + getpid() + getDevRandom());
		seedCount = 1;
	}
	return uniformRandGen.random();
}

inline int randInteger(int minRange, int maxRange)
{
    // get a random sample between minRange and maxRange !!!
    maxRange++;
    return min(maxRange-1, (int)(minRange + (double)(maxRange - minRange) * randDouble()));
}

inline VectorXi randInteger(const int& minRange, const int& maxRange, const int& numSample)
{
    // get values between minRange and maxRange !!!
	// min return = minRange
	// max return = maxRange (so the maxRange value is included!!!)
    VectorXi out(numSample);
    int range = maxRange - minRange;
    for (int nSamp = 0; nSamp < numSample; nSamp++)
        out(nSamp) = min(maxRange, (int)(minRange + (double)(range+1) * randDouble()));
    return out;
}


inline vector<int> randPermSTL(const int& num, const int numSample = -1)
{
	vector<pair<double, int> > pairs(num);
    pair<double, int> tmpPair;

    for (int n = 0; n < num; n++)
    {
        tmpPair.first = randDouble();
        tmpPair.second = n;
        //pairs.push_back(tmpPair);
        pairs[n] = tmpPair;
    }
    sort(pairs.begin(), pairs.end());

    int return_num = num;
    if (numSample > 0 && numSample < num)
    	return_num = numSample;
    std::vector<int> outRand(return_num);
	for (int n = 0; n < return_num; n++)
        outRand[n] = pairs[n].second;

    return outRand;
}


//! Returns the time (ms) elapsed between two calls to this function
inline double timeIt(int reset)
{
    static time_t startTime, endTime;
    static int timerWorking = 0;

    if (reset)
    {
        startTime = time(NULL);
        timerWorking = 1;
        return -1;
    }
    else
    {
        if (timerWorking)
        {
            endTime = time(NULL);
            timerWorking = 0;
            return (double) (endTime - startTime);
        }
        else
        {
            startTime = time(NULL);
            timerWorking = 1;
            return -1;
        }
    }
}

inline void fillWithRandomNumbers(std::vector<double>& tmpWeights)
{

  std::vector<double>::iterator it(tmpWeights.begin()), end(tmpWeights.end());
  for(; it != end;it++){
     *it = 2.0*(randDouble() - 0.5);
  }
}

inline void fillWithRandomNumbers(VectorXd& num_vector)
{
	int num_values = num_vector.rows();
	for (int v = 0; v < num_values; v++)
		num_vector(v) = 2.0 * (randDouble() - 0.5);
}


inline int samplePoissonKnuth(double lambda)
{
	double L = std::exp(-lambda);
	int k = 0;
	double p = 1.0;
	do
	{
		k++;
		p *= randDouble();
	}
	while (p > L);
	return k-1;
}

inline int samplePoissonAmir(double A)
{
    int k = 0;
    int maxK = 10;
    while (1) {
        double U_k = randDouble();
        A *= U_k;
        if (k > maxK || A < exp(-1.0)) {
            break;
        }
        k++;
    }
    return k;
}

inline double roundPrecision(double x, double precision)
{
	double multiplier = std::pow(10.0, precision-1.0);
	return floor(x*multiplier + 0.5)/multiplier;
}


inline void removeRowLibEigenMatrixXd(Eigen::MatrixXd& mat, unsigned int rowToRemove)
{
	// check input
	if (rowToRemove < 0 || rowToRemove >= mat.rows())
		throw std::runtime_error("Index out of range!");
	// new sizes
    unsigned int numRows = mat.rows()-1;
    unsigned int numCols = mat.cols();
    // remove row (if not last row)
    if (rowToRemove < numRows)
        mat.block(rowToRemove,0,numRows-rowToRemove,numCols) = mat.block(rowToRemove+1,0,numRows-rowToRemove,numCols);
    // resize matrix
    mat.conservativeResize(numRows,numCols);
}

inline void removeColumnLibEigenMatrixXd(Eigen::MatrixXd& mat, unsigned int colToRemove)
{
	// check input
	if (colToRemove < 0 || colToRemove >= mat.cols())
		throw std::runtime_error("Index out of range!");
	// new sizes
    unsigned int numRows = mat.rows();
    unsigned int numCols = mat.cols()-1;
    // remove col (if not last col)
    if (colToRemove < numCols)
        mat.block(0,colToRemove,numRows,numCols-colToRemove) = mat.block(0,colToRemove+1,numRows,numCols-colToRemove);
    // resize matrix
    mat.conservativeResize(numRows,numCols);
}

inline void pushbackColumnLibEigenMatrixXd(Eigen::MatrixXd& mat, Eigen::MatrixXd newCol)
{
	// if the matrix is of size (0,0) and we want to add the first column,
	// we have the set the appropriate number of rows!
	unsigned int numRows = mat.rows();
	if (mat.rows() == 0)
		mat.conservativeResize(newCol.rows(), Eigen::NoChange);

	unsigned int numCols = mat.cols();
	mat.conservativeResize(Eigen::NoChange, numCols+1);
	mat.col(numCols) = newCol;
}

#endif /* UTILITIES_H_ */
