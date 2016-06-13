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
AdaBoost::AdaBoost(vector<Data*>& data, int iterations) :
	iterations(iterations),
	features(data),
	strongClassifier(new StrongClassifier(vector<WeakClassifier*>{})){
	int size = features.size();
	cout << "Initializing AdaBoost with " << iterations << " iterations" << endl;
	cout << "Training size: " << size << "\n" << endl;
	//Initialize weights
	for(int m = 0; m < features.size(); ++m){
		features[m]->setWeight((float) 1/features.size());
	}
	cout << "Initialized uniform weights\n" << endl;
}

AdaBoost::AdaBoost(): iterations(0),
		strongClassifier(new StrongClassifier(vector<WeakClassifier*> {})) {
}

/**
 * Train the AdaBoost classifier with a number of weak classifier specified
 * with the iteration attribute.
 */
StrongClassifier* AdaBoost::train(vector<WeakClassifier*>& classifiers){
	cout << "Training AdaBoost with " << iterations << " iterations" << endl;
	auto t_start = chrono::high_resolution_clock::now();

	//Iterate for the specified iterations
	for (unsigned int i = (iterations - (iterations - classifiers.size())); i < iterations; ++i) {
		cout << "Iteration: " << (i + 1) << endl;;
		WeakClassifier* weakClassifier = trainWeakClassifier();
		float error = weakClassifier->getError();
		if(error < 0.5){
			float alpha = updateAlpha(error);
			float beta = updateBeta(error);
			weakClassifier->setAlpha(alpha);
			weakClassifier->setBeta(beta);
			updateWeights(weakClassifier);
			normalizeWeights();
			weakClassifier->printInfo();
			classifiers.push_back(weakClassifier);
			//If error is 0, classification is perfect (linearly separable data)
			if(error == 0){
				break;
			}
		} else {
			cout << "Error: weak classifier with error > 0.5." << endl;
			break;
		}
	}
	//showFeatures();
	//Create strong classifier
	strongClassifier->setClassifiers(classifiers);

    auto t_end = high_resolution_clock::now();
    cout << "Time: " << (duration<double, milli>(t_end - t_start).count())/1000 << " s" << endl;
    return strongClassifier;
}

StrongClassifier* AdaBoost::train(){
	//The vector of weak classifiers
	vector<WeakClassifier*> classifiers;
	return train(classifiers);
}

int AdaBoost::predict(Data* x){
	return strongClassifier->predict(x);
}

/**
 * Updates features weights according to their errors
 * Weights of training examples misclassified are increased by ht (x) and
 * weights of the examples correctly classified are decreased by ht (x) .
 * In this way, AdaBoost focuses on the most informative or difficult examples.
 */
void AdaBoost::updateWeights(WeakClassifier* weakClassifier){
	for(int i = 0; i < features.size(); ++i){
		float num = (features[i]->getWeight() * exp(-weakClassifier->getAlpha()
				* features[i]->getLabel() * weakClassifier->predict(this->features[i])));
		features[i]->setWeight(num);
	}
}

void AdaBoost::normalizeWeights(){
	float norm = 0;
	for(int i = 0; i < features.size(); ++i){
		norm += features[i]->getWeight();
	}
	for(int i = 0; i < features.size(); ++i){
		//Normalize such that wt+1 is a prob. distribution
		features[i]->setWeight((float) features[i]->getWeight()/norm);
	}
}

/***
 * Train weak classifier on training data choosing the one minimizing the error
 */
WeakClassifier* AdaBoost::trainWeakClassifier(){
	WeakClassifier* bestWeakClass = new WeakClassifier();

	if (features.size() > 0) {
		//Feature vector dimension
		int dimensions = features[0]->getFeatures().size();
		//Error and signs vector
		vector<example> signs;
		vector<float> errors;
		vector<int> misclassifies;
		//Cumulative sums of the weights
		float posWeights = 0;
		float negWeights = 0;
		float totNegWeights = 0;
		float totPosWeights = 0;
		//Number of examples
		int totPositive = 0;
		int totNegative = 0;
		int cumPositive = 0;
		int cumNegative = 0;
		//Errors
		float weight, error;
		float errorPos, errorNeg;
		float threshold;
		int index;

		//Evaluating total sum of negative and positive weights
		for (unsigned int i = 0; i < features.size(); ++i) {
			if (features[i]->getLabel() == 1) {
				totPosWeights += features[i]->getWeight();
				totPositive++;
			} else {
				totNegWeights += features[i]->getWeight();
				totNegative++;
			}
		}

		//Iterate through dimensions
		for (unsigned int j = 0; j < dimensions; ++j) {

			//Sorts vector of features according to the j-th dimension
			sort(features.begin(), features.end(),
					[j](Data* const &a, Data* const &b) {return a->getFeatures()[j] < b->getFeatures()[j];});

			//Reinitialize variables
			signs.clear();
			errors.clear();
			misclassifies.clear();

			posWeights = 0;
			negWeights = 0;
			cumNegative = 0;
			cumPositive = 0;

			//Iterates features
			for (int i = 0; i < features.size(); ++i) {
				weight = features[i]->getWeight();
				if (features[i]->getLabel() == 1) {
					posWeights += weight;
					cumPositive++;
				} else {
					negWeights += weight;
					cumNegative++;
				}

				errorPos = posWeights + (totNegWeights - negWeights);
				errorNeg = negWeights + (totPosWeights - posWeights);

				if ( (i < features.size() - 1 && features[i]->getFeatures()[j] != features[i + 1]->getFeatures()[j]) || i == features.size() - 1 ){
					if (errorPos > errorNeg) {
						errors.push_back(errorNeg);
						signs.push_back(POSITIVE);
						misclassifies.push_back(
								cumNegative + (totPositive - cumPositive));
					} else {
						errors.push_back(errorPos);
						signs.push_back(NEGATIVE);
						misclassifies.push_back(
								cumPositive + (totNegative - cumNegative));
					}
				} else {
					errors.push_back(1.);
					signs.push_back(POSITIVE);
					misclassifies.push_back(0);
				}

			}

			auto errorMin = min_element(begin(errors), end(errors));
			error = *errorMin;

			if (error < bestWeakClass->getError()) {
				index = errorMin - errors.begin();
				threshold = (features[index])->getFeatures()[j];
				bestWeakClass->setError(error);
				bestWeakClass->setDimension(j);
				bestWeakClass->setThreshold(threshold);
				bestWeakClass->setMisclassified(misclassifies[index]);
				bestWeakClass->setSign(signs[index]);
			}

			cout << "\rEvaluated: " << j + 1 << "/" << dimensions << " features" << flush;
		}
	}
	return bestWeakClass;
}

float AdaBoost::updateAlpha(float error){
	return  0.5 * log((1 - error) / error);
}

float AdaBoost::updateBeta(float error){
	return error / (1 - error);
}

void AdaBoost::showFeatures(){
	for(int i = 0; i < features.size(); ++i){
		features[i]->print();
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
