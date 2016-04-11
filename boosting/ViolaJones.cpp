/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.h"

ViolaJones::ViolaJones(vector<Data> data, vector<double> weights, int iterations):
	AdaBoost(data, weights, iterations){
	this->maxInterations = iterations;
}

void ViolaJones::normalizeWeights(){
	double norm = 0;
	for (int i = 0; i < features.size(); ++i) {
		norm += features[i].getWeight();
	}
	for (int i = 0; i < features.size(); ++i) {
		features[i].setWeight((double) features[i].getWeight() / norm);
	}
}

void ViolaJones::updateWeights(WeakClassifier* weakClassifier){
	for(int i = 0; i < features.size(); ++i){
		int e = (features[i].getLabel()
				* weakClassifier->predict(this->features[i]) > 0) ? 0 : 1;
		double num = features[i].getWeight() * (pow(weakClassifier->getBeta(), (double) (1 - e)));
		features[i].setWeight(num);
	}
}

void ViolaJones::train(){
	//TODO will be two attributes
	double targetFPR = 0.3;
	double targetFPRLayer = 0.5;

	double F = 1.0;
	double Fold = 1.0;
	double D = 1.0;
	int i = 0;
	int n = 0;
	while(F > targetFPR){
		i++;
		n = 0;
		F = Fold;
		while(F > targetFPRLayer * Fold){
			n++;
			this->iterations = n;
			AdaBoost::train();
		    //Evaluate current cascaded classifier on validation set to determine F(i) & D(i)
			pair<double, double> rates = computeRates(features);
			F = rates.first;
			D = rates.second;
//			decrease threshold for the ith classifier
//			        until the current cascaded classifier has a detection rate of at least d x D(i-1) (this also affects F(i))

//			 N = ∅
//			if F(i) > Ftarget then
//			evaluate the current cascaded detector on the set of non-face images
//			and put any false detections into the set N.

		}

//		while Fi > f × Fi−1
//		∗ni ←ni +1
//		∗ Use P and N to train a classifier with ni features using
//		AdaBoost
//		∗ Evaluate current cascaded classifier on validation set to
//		determine Fi and Di .
//		∗ Decrease threshold for the ith classifier until the current
//		cascaded classifier has a detection rate of at least d × Di −1 (this also affects Fi )

	}

}

pair<double, double> ViolaJones::computeRates(vector<Data> features){
	pair<double, double> output;
	vector<int> predictions = strongClassifier.predict(features);
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	for(int i = 0; i < predictions.size(); ++i){
		if(predictions[i] == 1 && features[i].getLabel() == -1){
			fp++;
		}
		if(predictions[i] == -1 && features[i].getLabel() == -1){
			tn++;
		}
		if(predictions[i] == -1 && features[i].getLabel() == 1){
			fn++;
		}
		if(predictions[i] == 1 && features[i].getLabel() == 1){
			tp++;
		}
	}
	output.first = (double) fp / (fp + tn);
	output.second = (double) tp / (tp + fn);
	return output;
}

ViolaJones::~ViolaJones(){}

