/*
 * Stage.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "Stage.h"

Stage::Stage(int number):
    number(number), classifiers({}), fpr(1.), detectionRate(1.), threshold(0.){
	cout << "\n*** Stage n. " << number << " ***\n" << endl;
}

Stage::Stage(int number, vector<WeakClassifier> weaks):
    number(number), classifiers(weaks), fpr(1.), detectionRate(1.){
	threshold = 0;
	for(int i = 0; i < classifiers.size(); ++i){
		threshold += classifiers[i].getAlpha();
	}
	threshold = threshold * 0.5;
}

int Stage::predict(Data x){
	double sum = 0;
	for(int i = 0; i < classifiers.size(); ++i){
		sum += classifiers[i].getAlpha() * classifiers[i].predict(x);
		//cout << "class: " << classifiers[i]->predict(x)<< endl;
	}


	return sum >= threshold ? 1 : -1;
}

void Stage::decreaseThreshold(double value){
	threshold -= value;
}

double Stage::getThreshold() const {
	return threshold;
}

void Stage::setThreshold(double threshold) {
	this->threshold = threshold;
}

Stage::~Stage() {
}

double Stage::getDetectionRate() const {
	return detectionRate;
}

void Stage::setDetectionRate(double detectionRate) {
	this->detectionRate = detectionRate;
}

double Stage::getFpr() const {
	return fpr;
}

void Stage::setFpr(double fpr) {
	this->fpr = fpr;
}

int Stage::getNumber() const {
	return number;
}

void Stage::setClassifiers(const vector<WeakClassifier>& classifiers) {
	this->classifiers = classifiers;
	threshold = 0;
	for (int i = 0; i < classifiers.size(); ++i) {
		threshold += classifiers[i].getAlpha();
	}
	threshold = threshold * 0.5;
}

void Stage::setNumber(int number) {
	this->number = number;
}

void Stage::printInfo(){
	cout << "\nStage n. " << number << ", FPR: " << fpr << ", DR: " << detectionRate << endl;
}

const vector<WeakClassifier>& Stage::getClassifiers() const {
	return classifiers;
}

