/*
 * MultiWeakClassifier.cpp
 *
 *  Created on: 21/mag/2016
 *      Author: lorenzocioni
 */

#include "MultiWeakClassifier.h"

MultiWeakClassifier::MultiWeakClassifier(int classes): WeakClassifier(){
	this->classes = classes;
}

int MultiWeakClassifier::predict(Data* x){
	return predict(x->getFeatures());
}

int MultiWeakClassifier::predict(vector<double> x){
	return 1;
}

MultiWeakClassifier::~MultiWeakClassifier(){}


