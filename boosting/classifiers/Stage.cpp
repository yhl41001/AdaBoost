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
	return sum >= threshold ? 1 : -1;
}

int Stage::predict(Mat img){
	double sum = 0;
	double value;
	for (int i = 0; i < classifiers.size(); ++i) {
		value = HaarFeatures::evaluate(img, classifiers[i]->getWhites(), classifiers[i]->getBlacks());
		sum += classifiers[i]->getAlpha() * classifiers[i]->predict(value);
	}
	return sum >= threshold ? 1 : -1;
}

void Stage::optimizeThreshold(vector<Data*> &positiveSet, double maxfnr){
	int wf;
	float thr;
    float *scores = new float[positiveSet.size()];
	for (int i=0; i< positiveSet.size(); i++) {
		scores[i] = 0;
	    wf = 0;
	    for (vector<WeakClassifier*>::iterator it = classifiers.begin(); it != classifiers.end(); ++it, wf++)
	      scores[i] += (*it)->getBeta() * ((*it)->predict(positiveSet[i]));
	  }
	  sort(scores, scores + positiveSet.size());
	  int maxfnrind = maxfnr * positiveSet.size();
	  if (maxfnrind >= 0 && maxfnrind < positiveSet.size()) {
	    thr = scores[maxfnrind];
	    while (maxfnrind > 0 && scores[maxfnrind] == thr) maxfnrind--;
	    threshold = scores[maxfnrind];
	  }
	  delete[] scores;
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
	cout << "\nStage n. " << number << ", FPR: " << fpr << ", DR: " << detectionRate << endl;
}

const vector<WeakClassifier*>& Stage::getClassifiers() const {
	return classifiers;
}

void Stage::addClassifier(WeakClassifier* wc){
	classifiers.push_back(wc);
}

Stage::~Stage() {
}

