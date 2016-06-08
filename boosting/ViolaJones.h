/*
 * ViolaJones.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_VIOLAJONES_H_
#define BOOSTING_VIOLAJONES_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "AdaBoost.h"
#include "classifiers/StrongClassifier.h"
#include "classifiers/CascadeClassifier.h"
#include "features/HaarFeatures.h"
#include "utils/Face.h"
#include "utils/Utils.hpp"

using namespace std;
using namespace cv;

class ViolaJones: public AdaBoost {
private:
	int maxStages;
	vector<Data*> positives;
	vector<Data*> negatives;
	vector<Data*> validation;
	string positivePath;
	string negativePath;
	string validationPath;
	int numPositives;
	int numNegatives;
	CascadeClassifier classifier;
	int negativesPerLayer;
	int detectionWindowSize;
	double evaluateFPR(vector<Data*> validationSet);
	double evaluateDR(vector<Data*> validationSet);
	void initializeWeights();
	void generateNegativeSet(bool newExamples);
	void extractFeatures();


protected:
	double updateAlpha(double error);
	double updateBeta(double error);
	void normalizeWeights();
	void updateWeights(WeakClassifier* weakClassifier);

public:
	ViolaJones();
	ViolaJones(string trainedPath);
	ViolaJones(string positivePath, string negativePath, int maxStages, int numPositives,
			int numNegatives, int detectionWindowSize = 24, int negativesPerLayer = 0);
	vector<Face> mergeDetections(vector<Face> detections, int padding = 6, double th = 0.5);
	void train();
	int predict(Mat img);
	void loadTrainedData(string filename);
	const string& getValidationPath() const;
	void setValidationPath(const string& validationPath);
	void store();
	~ViolaJones();
	const CascadeClassifier& getClassifier() const;
	void setClassifier(const CascadeClassifier& classifier);
};

#endif /* BOOSTING_VIOLAJONES_H_ */
