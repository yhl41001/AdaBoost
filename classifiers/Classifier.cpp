/*
 * Classifier.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "Classifier.h"

double Classifier::getError() const {
	return error;
}

void Classifier::setError(double error) {
	this->error = error;
}
