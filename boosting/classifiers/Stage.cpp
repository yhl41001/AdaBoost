/*
 * Stage.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "Stage.h"

Stage::Stage(int number):
    number(number), classifiers({}), fpr(1.), detectionRate(1.), threshold(0.){
}

Stage::Stage(int number, vector<WeakClassifier*> weaks):
    number(number), classifiers(weaks), fpr(1.), detectionRate(1.){
	threshold = 0;
	for(int i = 0; i < classifiers.size(); ++i){
		threshold += classifiers[i]->getAlpha();
	}
	threshold = threshold * 0.5;
}

int Stage::predict(const vector<double>& x){
	double sum = 0;
	for (int i = 0; i < classifiers.size(); ++i) {
		sum += classifiers[i]->getAlpha() * classifiers[i]->predict(x);
	}
	return sum >= threshold ? 1 : 0;
}

int Stage::predict(Mat img){
	double sum = 0;
	double value;
	for (int i = 0; i < classifiers.size(); ++i) {
		value = HaarFeatures::evaluate(img, classifiers[i]->getWhites(), classifiers[i]->getBlacks());
		sum += classifiers[i]->getAlpha() * classifiers[i]->predict(value);
	}
	return sum >= threshold ? 1 : 0;
}


void Stage::optimizeThreshold(vector<Data*>& positiveSet, double dr){
	cout << "Optimizing threshold for stage" << endl;
	vector<double> scores(positiveSet.size());
	double thr;
	for(int i = 0; i < positiveSet.size(); ++i){
		scores[i] = 0;
		for(int j = 0; j < classifiers.size(); ++j){
			scores[i] += classifiers[j]->getAlpha() * classifiers[j]->predict(positiveSet[i]);
		}
	}
	sort(scores.begin(), scores.end());
	int index = positiveSet.size() - dr * positiveSet.size();
	if(index >= 0 && index < positiveSet.size()){
		thr = scores[index];
		if(thr == 0){
			while(index < positiveSet.size() - 1 && scores[index] == 0){
				index++;
			}
			thr = scores[index];
		}
		threshold = scores[index];
	}
	cout << "Setting threshold to " << threshold << endl;
	scores.clear();
}

void Stage::decreaseThreshold(){
	double value = 1;
	threshold -= value;
	cout << "Decrease threshold to " << threshold << endl;
}

double Stage::getThreshold() const {
	return threshold;
}

void Stage::setThreshold(double threshold) {
	this->threshold = threshold;
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

void Stage::setClassifiers(const vector<WeakClassifier*>& classifiers) {
	this->classifiers = classifiers;
	threshold = 0;
	for (int i = 0; i < classifiers.size(); ++i) {
		threshold += classifiers[i]->getAlpha();
	}
	threshold = threshold * 0.5;
}

void Stage::setNumber(int number) {
	this->number = number;
}

void Stage::printInfo(){
	cout << "\nStage n. " << number << ", FPR: " << fpr << ", DR: " << detectionRate << ", Threshold: " << threshold << endl;
}

const vector<WeakClassifier*>& Stage::getClassifiers() const {
	return classifiers;
}

void Stage::addClassifier(WeakClassifier* wc){
	classifiers.push_back(wc);
}

Stage::~Stage() {
}

