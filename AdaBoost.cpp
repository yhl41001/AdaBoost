/*
 * AdaBoost.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 *
 *      Decision stump: Single axis-parallel partition of space
 */

#include "AdaBoost.h"

using namespace std;

AdaBoost::AdaBoost(vector<Feature> features, vector<int> labels, int iterations) :
	iterations(iterations),
	features(features),
	labels(labels){

	if(features.size() == labels.size()){
		int size = features.size();
		cout << "Initializing AdaBoost with " << iterations << " iterations" << endl;
		cout << "Training size: " << size << endl;
	} else {
		cout << "Error: features and labels must be in equal number." << endl;
	}
}

int AdaBoost::getIterations() const {
	return iterations;
}

void AdaBoost::setIterations(int iterations) {
	this->iterations = iterations;
}

void AdaBoost::train(){
	//Initialize weights
	int n = this->features.size();
	this->weights = vector<double>(n, (double) 1/n);

	//Iterate for the specified iterations
	for (int i = 0; i < this->iterations; ++i) {
		WeakClassifier* weakClassifier = trainWeakClassifier();
		double error = weakClassifier->getError();
		if(error < 0.5){
			double alpha = 1/2 * log((1 - error)/error);
			weakClassifier->setAlpha(alpha);
			updateWeights(weakClassifier);

			weakClassifier->printInfo();
		} else {
			cout << "stop" << endl;
		}
	}
}

void AdaBoost::updateWeights(WeakClassifier* weakClassifier){
	for(int i = 0; i < this->features.size(); ++i){
		double num = (this->weights[i] * exp(-weakClassifier->getAlpha()
				* this->labels[i] * weakClassifier->predict(this->features[i])));
		double normalisation = 1;
		//Normalize such that wt+1 is a prob. distribution
		this->weights[i] = num/normalisation;
	}
}

WeakClassifier* AdaBoost::trainWeakClassifier(){

	for(int j = 0; j < this->features.size(); ++j){
				this->features[j].print();
			}

	WeakClassifier* weakClassifier = new WeakClassifier();
	if(this->features.size() > 0){
		//Feature vector dimension
		int size = this->features[0].getFeatures().size();
		//Iterate through dimensions
		for(int i = 0; i < size; ++i){

			sort(this->features.begin(), this->features.end(), FeatureComparator(i));
		}

		for(int j = 0; j < this->features.size(); ++j){
			this->features[j].print();
		}


	}


	//TODO training of classifier
	return weakClassifier;
}

AdaBoost::~AdaBoost(){
	weights.clear();
	features.clear();
	labels.clear();
	cout << "Removing AdaBoost from memory" << endl;
}



