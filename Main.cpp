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
#include "boosting/utils/Utils.hpp"
#include "facedetector/FaceDetector.h"
#include "digitclassifier/DigitsClassifier.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagePath = "/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/dataset/";

	Utils::generateNonFacesDataset(imagePath + "backgrounds/", imagePath + "negatives", 9758, 24);
	string positivePath = imagePath + "lfwcrop/faces/";
	string negativePath = imagePath + "negatives/";
	string validationPath = imagePath + "validation/";


	//Mat test = imread(imagePath + "test/tammy.jpg", 0);
	//Mat test = imread(imagePath + "test/knex0.jpg", 0);
	//Mat test = imread(imagePath + "lfwcrop/faces/Ana_Isabel_Sanchez_0001.pgm", 0);


	//FaceDetector* detector = new FaceDetector(positivePath, negativePath, 24, 2000, 3000);
	//detector->setValidationPath(validationPath);
	//detector->train();

//	FaceDetector* detector = new FaceDetector("trainedDataOld.txt", 8);
//	detector->displaySelectedFeatures(test, 2);

//	detector->detect(test, true);


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
