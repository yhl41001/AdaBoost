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

/**
 * Initialize a new adaboost object with a vector of training samples (features)
 * and a given number of iterations
 */
AdaBoost::AdaBoost(vector<Feature> data, int iterations) :
	iterations(iterations),
	features(data),
	strongClassifier(*(new StrongClassifier(vector<WeakClassifier>{}))){
	int size = features.size();
	cout << "Initializing AdaBoost with " << iterations << " iterations" << endl;
	cout << "Training size: " << size << "\n" << endl;
}

/**
 * Train the AdaBoost classifier with a number of weak classifier specified
 * with the iteration attributes.
 */
void AdaBoost::train(){
	clock_t c_start = clock();
	auto t_start = chrono::high_resolution_clock::now();

	//Reinitialize classifier
	strongClassifier.setTrained(false);

	//Initialize weights
	for(int m = 0; m < features.size(); ++m){
		features[m].setWeight((double) 1/features.size());
	}

	//The vector of weak classifiers
	vector<WeakClassifier> classifiers;

	//Iterate for the specified iterations
	for (int i = 0; i < this->iterations; ++i) {
		cout << "Iteration: " << (i + 1) << " | ";
		WeakClassifier* weakClassifier = trainWeakClassifier();
		double error = weakClassifier->getError();
		if(error < 0.5){
			double alpha = 0.5 * log((1 - error)/error);
			weakClassifier->setAlpha(alpha);
			updateWeights(weakClassifier);
			weakClassifier->printInfo();
			classifiers.push_back(*weakClassifier);
			//If error is 0, classification is perfect (lineraly separable data)
			if(error == 0){
				break;
			}
		} else {
			cout << "Error: weak classifier with error > 0.5." << endl;
		}
	}
	//showFeatures();
	//Create strong classifier
	strongClassifier.setClassifiers(classifiers);
	strongClassifier.setTrained(true);

    clock_t c_end = clock();
    auto t_end = chrono::high_resolution_clock::now();

    cout << std::fixed << "\nCPU time used: "
         << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n"
         << "Time: "
         << chrono::duration<double, milli>(t_end - t_start).count()
         << " ms\n\n";
}

int AdaBoost::predict(Feature x){
	if(strongClassifier.isTrained()){
		return strongClassifier.predict(x);
	} else {
		cout << "The classifier is not trained. Please train the classifier first." << endl;
		return 0;
	}
}

/**
 * Updates features weights according to their errors
 * Weights of training examples misclassified are increased by ht (x) and
 * weights of the examples correctly classified are decreased by ht (x) .
 * In this way, AdaBoost focuses on the most informative or difficult examples.
 */
void AdaBoost::updateWeights(WeakClassifier* weakClassifier){
	double norm = 0;
	for(int i = 0; i < features.size(); ++i){
		double num = (features[i].getWeight() * exp(-weakClassifier->getAlpha()
				* features[i].getLabel() * weakClassifier->predict(this->features[i])));
		norm += num;
		features[i].setWeight(num);
	}
	for(int i = 0; i < features.size(); ++i){
		//Normalize such that wt+1 is a prob. distribution
		features[i].setWeight((double) features[i].getWeight()/norm);
	}
}

/***
 * Train weak classifier on training data choosing the one minimizing the error
 */
WeakClassifier* AdaBoost::trainWeakClassifier(){
	WeakClassifier* bestWeakClass = new WeakClassifier();

	if(features.size() > 0){
		//Feature vector dimension
		int size = features[0].getFeatures().size();

		//Cumulative sums
		vector<double> w;

		//Iterate through dimensions
		for(int j = 0; j < size; ++j){
			WeakClassifier* weakClassifier = new WeakClassifier();
			weakClassifier->setDimension(j);

			//Sorts vector of features according to the j-th dimension
			sort(features.begin(), features.end(), FeatureComparator(j));

			w.clear();
			double sum = 0;

			//Iterates features
			for(int i = 0; i < features.size(); ++i){
				sum = sum + features[i].getWeight() * features[i].getLabel();
				w.push_back(sum);
			}

			//Retrieving min and max of the sums
			auto result = minmax_element(w.begin(), w.end());
			double min = w[result.first - w.begin()];
			double max = w[result.second - w.begin()];
			int index;
			double threshold;

			if(abs(min) > abs(max)){
				//Negative values
				index = result.first - w.begin();

			} else {
				//Positive values
				index = result.second - w.begin();
			}

			if(w[index] > 0){
				weakClassifier->setSign(POSITIVE);
			} else {
				weakClassifier->setSign(NEGATIVE);
			}

			//Setting threshold
			threshold = (features[index]).getFeatures()[j];
			weakClassifier->setThreshold(threshold);

			double error = weakClassifier->evaluateError(features);
			if(error < bestWeakClass->getError()){
				bestWeakClass->setError(error);
				bestWeakClass->setDimension(j);
				bestWeakClass->setThreshold(threshold);
				bestWeakClass->setMisclassified(weakClassifier->getMisclassified());
				bestWeakClass->setSign(weakClassifier->getSign());
			}
		}
	}
	return bestWeakClass;
}


void AdaBoost::showFeatures(){
	for(int i = 0; i < features.size(); ++i){
		features[i].print();
	}
}

int AdaBoost::getIterations() const {
	return iterations;
}

void AdaBoost::setIterations(int iterations) {
	this->iterations = iterations;
}

/**
 * Deconstructor: free memory
 */
AdaBoost::~AdaBoost(){
	features.clear();
	cout << "Removing AdaBoost from memory" << endl;
}
