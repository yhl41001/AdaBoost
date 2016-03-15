/*
 * Classifier.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "WeakClassifier.h"

WeakClassifier::WeakClassifier(): error(0.), dimension(0), threshold(0.), alpha(0.){}

int WeakClassifier::predict(Feature x){
	//TODO implement prediction
	return 1;
}

/**
 * Print info about weak classifier
 */
void WeakClassifier::printInfo(){
	std::cout << "Alpha: " << alpha << ", Dimension: " << dimension
			<< ", Error: " << error << std::endl;
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

void WeakClassifier::setAlpha(double alpha) {
	this->alpha = alpha;
}
