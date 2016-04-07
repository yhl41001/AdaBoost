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
#include "facedetector/utils/IntegralImage.h"
#include "facedetector/features/HaarFeatures.h"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagesPath = "/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/images/2002/07/19/big/img_130.jpg";

	Mat img = imread(imagesPath, IMREAD_GRAYSCALE);

	IntegralImage* intImage = new IntegralImage(img);

	Mat subwindow = img(Rect(0, 0, 23, 23));

	HaarFeatures* haar = new HaarFeatures();
	haar->extractFeatures(*intImage, 150, 200);


	FaceDetector* detector = new FaceDetector(12);
	detector->train();


	//double a = intImage->computeArea(Rect(1, 1, 1, 1));
	//cout << a <<endl;

	cout << "AdaBoost classifier" << endl;

	vector<Data> features = {
		*(new Data(vector<double>{2, 2}, 1)),
		*(new Data(vector<double>{3, 5}, 1)),
		*(new Data(vector<double>{6, 2}, 1)),
		*(new Data(vector<double>{6, 6}, 1)),
		*(new Data(vector<double>{8, 4}, 1)),
		*(new Data(vector<double>{4, 3}, -1)),
		*(new Data(vector<double>{4, 4}, -1)),
		*(new Data(vector<double>{5, 3}, -1)),
		*(new Data(vector<double>{5, 4}, -1))
	};

	//AdaBoost* boost = new AdaBoost(features, 3);
	//boost->train();

	//int p = boost->predict(*(new Data(vector<double>{4.5, 3.5})));

	//cout << p << endl;
	//delete boost;
    return 0;
}
