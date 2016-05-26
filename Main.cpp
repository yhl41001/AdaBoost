/*
 * Main.cpp
 *
 *  Created on: 09/mar/2016
 *      Author: lorenzocioni
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "boosting/AdaBoost.h"
#include "boosting/features/Data.h"
#include "boosting/features/HaarFeatures.h"
#include "boosting/utils/IntegralImage.h"
#include "facedetector/FaceDetector.h"
#include "facedetector/utils/Utils.hpp"
#include "digitclassifier/DigitsClassifier.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagePath = "/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/dataset/";
	string path;

	vector<Mat> trainImages;
	vector<int> trainLabels;


	//Loading training positive images
	vector<string> positive = Utils::open(imagePath + "face");
	//int positiveExamples = positive.size();
	int positiveExamples = 100;
	for(int k = 0; k < positiveExamples; ++k){
		Mat img = imread(imagePath + "face/" + positive[k]);
		Mat dest;
		resize(img, dest, Size(24, 24));
		trainImages.push_back(dest);
		trainLabels.push_back(1);
	}

	//Loading training negative images
	vector<string> negativeImages = Utils::open(imagePath + "non-face");
	//int negativeExamples = negativeImages.size();
	int negativeExamples = 200;
	for(int k = 0; k < negativeExamples; ++k){
		Mat img = imread(imagePath + "non-face/" + negativeImages[k]);
		Mat dest;
		resize(img, dest, Size(24, 24));
		trainImages.push_back(dest);
		trainLabels.push_back(-1);
	}

	Mat test = imread(imagePath + "test/test.jpg", 0);
	//Mat test = imread(imagePath + "s1/1.pgm", 0);

	//FaceDetector* detector = new FaceDetector(trainImages, trainLabels, 8);
	//detector->train();

	FaceDetector* detector = new FaceDetector("trainedData.txt");
	detector->detect(test, true);


	/*
	string digitsPath = imagePath + "digits/train-images-idx3-ubyte";
	string digitsLabelsPath = imagePath + "digits/train-labels-idx1-ubyte";
	DigitsClassifier* digitsClassifier = new DigitsClassifier(digitsPath, digitsLabelsPath, 100);
	digitsClassifier->train();
	*/


	/*
	vector<Data*> features = {
		new Data(vector<double>{2, 2}, 1),
		new Data(vector<double>{2, 4}, 1),
		new Data(vector<double>{3, 8}, -1),
		new Data(vector<double>{2, 3}, -1),
		new Data(vector<double>{4, 9}, -1),
		new Data(vector<double>{6, 10}, -1),
		new Data(vector<double>{4, 5}, 1)
	};*/

	//AdaBoost* boost = new AdaBoost(features, 3);
	//boost->train();

	//int p = boost->predict(*(new Data(vector<double>{4.5, 3.5})));

	//cout << p << endl;
	//delete boost;

    return 0;
}
