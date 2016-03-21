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
#include "AdaBoost.h"
#include "features/Data.h"
#include "opencv2/highgui/highgui.hpp"
#include "utils/IntegralImage.h"


using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagesPath = "./images/2002/07/19/big/img_130.jpg";

	Mat img = imread(imagesPath, IMREAD_GRAYSCALE);

	IntegralImage* intImage = new IntegralImage(img);

	cout << "AdaBoost classifier" << endl;

	vector<Data> features = {
		*(new Data(vector<double>{1, 1}, 1)),
		*(new Data(vector<double>{1, 2}, 1))
//		*(new Feature(vector<double>{2, 1.5}, 1)),
//		*(new Feature(vector<double>{3, 2}, 1)),
//		*(new Feature(vector<double>{2.8, 4}, -1)),
//		*(new Feature(vector<double>{3.2, 1}, -1)),
//		*(new Feature(vector<double>{3, 3.5}, -1)),
//		*(new Feature(vector<double>{4, 1.5}, 1)),
//		*(new Feature(vector<double>{4.2, 4}, 1))
	};

	AdaBoost* boost = new AdaBoost(features, 20);
	boost->train();

	int p = boost->predict(*(new Data(vector<double>{5, 4})));

	cout << p << endl;
	delete boost;
    return 0;
}
