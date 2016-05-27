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
#include "features/Haar.h"
#include "classifiers/StrongClassifier.h"
#include "classifiers/CascadeClassifier.h"
#include "features/HaarFeatures.h"

using namespace std;
using namespace cv;

class ViolaJones: public AdaBoost {
private:
	int maxStages;
	vector<Data*> positives;
	vector<Data*> negatives;
	vector<Data*> falseDetections;
	vector<Haar> selectedFeatures;
	CascadeClassifier classifier;
	pair<double, double> computeRates(vector<Data*> validationSet);
	void initializeWeights();

protected:
	double updateAlpha(double error);
	double updateBeta(double error);
	void normalizeWeights();
	void updateWeights(WeakClassifier* weakClassifier);


public:
	ViolaJones();
	ViolaJones(string trainedPath);
	ViolaJones(vector<Data*> positives, vector<Data*> negatives, int maxStages);
	void train();
	int predict(Mat img, int size);
	int predict(vector<double> x);
	void loadTrainedData(string filename);
	void store();
	~ViolaJones();
	const vector<Haar>& getSelectedFeatures() const;
	void setSelectedFeatures(const vector<Haar>& selectedFeatures);
	vector<Rect> mergeDetections(vector<Rect> &detections);
};

#endif /* BOOSTING_VIOLAJONES_H_ */
