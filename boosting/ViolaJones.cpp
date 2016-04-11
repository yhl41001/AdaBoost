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
		Stage* stage;
		while(FPR > minFPR * FPRold){
			n++;
			this->iterations = n;
			StrongClassifier strongClassifier = AdaBoost::train();
		    //Evaluate current cascaded classifier on validation set to determine F(i) & D(i)
			pair<double, double> rates = computeRates(features);
			FPR = rates.first;
			DR = rates.second;
			stage = new Stage(i, strongClassifier.getClassifiers(), FPR, DR);
			//until the current cascaded classifier has a detection rate of at least d x D(i-1) (this also affects F(i))
			while(DR < minDR * DRold ){
				//decrease threshold for the ith classifier
				stage->decreaseThreshold(0.1);
				pair<double, double> rates = computeRates(features);
				FPR = rates.first;
				DR = rates.second;
			}


//			 N = ∅
//			if F(i) > Ftarget then
//			evaluate the current cascaded detector on the set of non-face images
//			and put any false detections into the set N.

		}


		cout << "stage added" << endl;

		FPRold = FPR;
		classifier.addStage(*stage);
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
		}
		if(predictions[i] == -1 && features[i].getLabel() == -1){
			tn++;
		}
		if(predictions[i] == -1 && features[i].getLabel() == 1){
			fn++;
		}
		if(predictions[i] == 1 && features[i].getLabel() == 1){
			tp++;
		}
	}

	cout << "FP: " << fp << " TN: " << tn << " FN: " << fn << "TP: " << tp << endl;
	output.first = (double) fp / (fp + tn);
	output.second = (double) tp / (tp + fn);

	cout << "FR: " << output.first << " DR: " << output.second << endl;
	return output;
}

ViolaJones::~ViolaJones(){}

