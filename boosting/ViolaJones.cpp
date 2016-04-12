/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.h"

ViolaJones::ViolaJones(vector<Data> positives, vector<Data> negatives, int iterations):
	AdaBoost(positives, negatives, iterations){
	this->maxInterations = iterations;
	this->classifier = *(new CascadeClassifier());
	this->positives = positives;
	this->negatives = negatives;
}

double ViolaJones::updateAlpha(double error){
	return  log((1 - error) / error);
}

double ViolaJones::updateBeta(double error){
	return error / (1 - error);
}

void ViolaJones::normalizeWeights(){
	double norm = 0;
	for (int i = 0; i < features.size(); ++i) {
		norm += features[i].getWeight();
	}
	for (int i = 0; i < features.size(); ++i) {
		features[i].setWeight((double) features[i].getWeight() / norm);
	}
}

void ViolaJones::updateWeights(WeakClassifier* weakClassifier){
	for(int i = 0; i < features.size(); ++i){
		int e = (features[i].getLabel()
				* weakClassifier->predict(this->features[i]) > 0) ? 0 : 1;
		double num = features[i].getWeight() * (pow(weakClassifier->getBeta(), (double) (1 - e)));
		features[i].setWeight(num);
	}
}

void ViolaJones::train(){
	//TODO will be two attributes
	double targetFPR = 0.3;
	double minFPR = 0.65;
	double minDR = 0.7;

	double FPR = 1.0;
	double FPRold = FPR;
	double DR = 1.0;
	double DRold = DR;

	int i = 0;
	int n = 0;
	while(FPR > targetFPR){
		i++;
		n = 0;
		FPR = FPRold;
		DR = DRold;
		Stage* stage = new Stage(i - 1);
		classifier.addStage(stage);
		while(FPR > minFPR * FPRold){
			n++;
			this->iterations = n;
			resetWeights();
			StrongClassifier strongClassifier = AdaBoost::train();
			stage->setClassifiers(strongClassifier.getClassifiers());
		    //Evaluate current cascaded classifier on validation set to determine F(i) & D(i)
			pair<double, double> rates = computeRates(features);
			FPR = rates.first;
			DR = rates.second;

			//until the current cascaded classifier has a detection rate of at least d x D(i-1) (this also affects F(i))
			while(DR < minDR * DRold ){
				//decrease threshold for the ith classifier
				cout << "decrease" << endl;
				stage->decreaseThreshold(0.1);
				pair<double, double> rates = computeRates(features);
				FPR = rates.first;
				DR = rates.second;
			}

			//N = ∅


			if(FPR > targetFPR){
				//if F(i) > Ftarget then
				//evaluate the current cascaded detector on the set of non-face images
				//and put any false detections into the set N.
			}
		}


		cout << "stage added" << endl;

		FPRold = FPR;
	}

}

pair<double, double> ViolaJones::computeRates(vector<Data> features){
	pair<double, double> output;
	vector<int> predictions = classifier.predict(features);
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	for(int i = 0; i < predictions.size(); ++i){
		if(predictions[i] == 1 && features[i].getLabel() == -1){
			fp++;
		} else if(predictions[i] == -1 && features[i].getLabel() == -1){
			tn++;
		} else if(predictions[i] == -1 && features[i].getLabel() == 1){
			fn++;
		} else if(predictions[i] == 1 && features[i].getLabel() == 1){
			tp++;
		}
	}

	cout << "FP: " << fp << ", TN: " << tn << ", FN: " << fn << ", TP: " << tp << endl;
	output.first = (double) fp / (fp + tn);
	output.second = (double) tp / (tp + fn);

	cout << "FPR: " << output.first << ", Detection Rate: " << output.second << endl;
	return output;
}

void ViolaJones::resetWeights(){
	for (int i = 0; i < features.size(); ++i) {
		/*	Initialize weights */
		if (features[i].getLabel() == 1) {
			features[i].setWeight((double) 1 / (2 * positives.size()));
		} else {
			features[i].setWeight((double) 1 / (2 * negatives.size()));
		}
	}
}

ViolaJones::~ViolaJones(){}
