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

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagePath = "/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/dataset/";

	//Utils::generateNonFacesDataset(imagePath + "backgrounds/", imagePath + "validation", 5000, 24);
	string positivePath = imagePath + "lfwcrop/faces/";
	string negativePath = imagePath + "backgrounds/";
	string validationPath = imagePath + "validation/";


	Mat test = imread(imagePath + "test/tammy.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread(imagePath + "test/knex0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread(imagePath + "lfwcrop/faces/Ana_Isabel_Sanchez_0001.pgm", CV_LOAD_IMAGE_GRAYSCALE);


	//FaceDetector* detector = new FaceDetector(positivePath, negativePath, 12, 3000, 3000);
	//detector->setValidationSet(validationPath);
	//detector->train();

	FaceDetector* detector = new FaceDetector("trainedData.txt", 1);
	//detector->displaySelectedFeatures(test);
	detector->detect(test, true);

	/*
	string digitsPath = imagePath + "digits/train-images-idx3-ubyte";
	string digitsLabelsPath = imagePath + "digits/train-labels-idx1-ubyte";
	DigitsClassifier* digitsClassifier = new DigitsClassifier(digitsPath, digitsLabelsPath, 100);
	digitsClassifier->train();
	*/

    return 0;
}
