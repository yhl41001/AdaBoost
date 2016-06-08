/*
 * Classifier.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "WeakClassifier.h"

WeakClassifier::WeakClassifier(): error(1.), dimension(0),
	threshold(0.), alpha(0.), beta(0.),
	sign(POSITIVE), misclassified(0){}

/**
 * Predict feature label
 */
int WeakClassifier::predict(Data* x){
	return predict(x->getFeatures()[dimension]);
}

int WeakClassifier::predict(const vector<double>& x){
	return predict(x[dimension]);
}

int WeakClassifier::predict(double value){
	if(value <= threshold){
		if (sign == POSITIVE)
			return 1;
		else
			return -1;
	} else {
		if (sign == POSITIVE)
			return -1;
		else
			return 1;
	}
}

/**
 * Evaluate error base on weights and misclassified samples
 */
double WeakClassifier::evaluateError(vector<Data*>& features){
	double error = 0;
	misclassified = 0;
	for(int i = 0; i < features.size(); ++i){
		int pred = predict(features[i]);
		if(pred != features[i]->getLabel()){
			error += features[i]->getWeight();
			misclassified += 1;
		}
	}
	return error;
}

/**
 * Print info about weak classifier
 */
void WeakClassifier::printInfo(){
	std::cout << "\nTrained WeakClassifier: alpha: " << alpha << ", dimension: " << dimension
			<< ", error: " << error << ", misclassified: " << misclassified
			<< ", threshold: " << threshold
			<< ", sign: ";
	if(sign == POSITIVE){
		std::cout << "positive" << std::endl;
	} else {
		std::cout << "negative" << std::endl;
	}
}

double WeakClassifier::getError() const {
	return error;
}

int WeakClassifier::getDimension() const {
	return dimension;
}

void WeakClassifier::setDimension(int dimension) {
	this->dimension = dimension;
}

double WeakClassifier::getThreshold() const {
	return threshold;
}

void WeakClassifier::setThreshold(double threshold) {
	this->threshold = threshold;
}

void WeakClassifier::setError(double error) {
	this->error = error;
}

double WeakClassifier::getAlpha() const {
	return alpha;
}

example WeakClassifier::getSign() const {
	return sign;
}

void WeakClassifier::setSign(example sign) {
	this->sign = sign;
}

void WeakClassifier::setAlpha(double alpha) {
	this->alpha = alpha;
	this->beta = exp(-alpha);
}

int WeakClassifier::getMisclassified() const {
	return misclassified;
}

void WeakClassifier::setMisclassified(int misclassified) {
	this->misclassified = misclassified;
}

double WeakClassifier::getBeta() const {
	return beta;
}

const vector<Rect>& WeakClassifier::getBlacks() const {
	return blacks;
}

void WeakClassifier::setBlacks(const vector<Rect>& blacks) {
	this->blacks = blacks;
}

const vector<Rect>& WeakClassifier::getWhites() const {
	return whites;
}

void WeakClassifier::setWhites(const vector<Rect>& whites) {
	this->whites = whites;
}

void WeakClassifier::setBeta(double beta) {
	this->beta = beta;
}
