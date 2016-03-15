/*
 * Classifier.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "WeakClassifier.h"

double WeakClassifier::getError() const {
	return error;
}

void WeakClassifier::setError(double error) {
	this->error = error;
}
