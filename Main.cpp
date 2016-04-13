/*
 * Main.cpp
 *
 *  Created on: 09/mar/2016
 *      Author: lorenzocioni
 */

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "boosting/AdaBoost.h"
#include "boosting/features/Data.h"
#include "facedetector/FaceDetector.h"
#include "facedetector/utils/IntegralImage.h"
#include "facedetector/features/HaarFeatures.h"
#include "facedetector/utils/Utils.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	int subjects = 10;
	int poses = 1;
	string imagePath = "/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/dataset/";
	string path;

	vector<Mat> trainImages;
	vector<int> trainLabels;

	//Loading training positive images
	for(int s = 1; s <= subjects; ++s){
		for(int p = 1; p <= poses; ++p){
			path = imagePath + "s" + to_string(s) + "/" + to_string(p) + ".pgm";
			Mat img = imread(path);
			Mat dest;
			resize(img, dest, Size(24, 24));
			trainImages.push_back(dest);
			trainLabels.push_back(1);
		}
	}

	//Loading training negative images
	vector<string> negativeImages = Utils::open(imagePath + "negative");
	int negativeExamples = 10;
	for(int k = 0; k < negativeExamples; ++k){
		Mat img = imread(imagePath + "negative/" + negativeImages[k]);
		Mat dest;
		resize(img, dest, Size(24, 24));
		trainImages.push_back(dest);
		trainLabels.push_back(-1);
	}

	FaceDetector* detector = new FaceDetector(trainImages, trainLabels, 12);
	detector->train();


	/*
	Mat test = imread(imagePath + "test/test.jpg", 0);
	imshow("img", test);
	waitKey(0);

	detector->detect(test);

	*/


	//double a = intImage->computeArea(Rect(1, 1, 1, 1));
	//cout << a <<endl;


	/*
	vector<Data> features = {
		*(new Data(vector<double>{2, 2}, 1)),
		*(new Data(vector<double>{2, 4}, 1)),
		*(new Data(vector<double>{2, 8}, -1)),
		*(new Data(vector<double>{2, 3}, -1))
		*(new Data(vector<double>{4, 9}, -1))
		*(new Data(vector<double>{6, 10}, -1)),
		*(new Data(vector<double>{4, 5}, 1))
	};

	AdaBoost* boost = new AdaBoost(features, 1);
	boost->train();*/

	//int p = boost->predict(*(new Data(vector<double>{4.5, 3.5})));

	//cout << p << endl;
	//delete boost;
    return 0;
}
