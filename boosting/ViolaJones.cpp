/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.h"

ViolaJones::ViolaJones(string trainedPath){
	//TODO load from file
}

ViolaJones::ViolaJones(vector<Data> positives, vector<Data> negatives, int maxStages, int maxIter):
	AdaBoost(){
	this->iterations = maxIter;
	this->maxInterations = maxIter;
	this->maxStages = maxStages;
	this->classifier = *(new CascadeClassifier());
	this->positives = positives;
	this->negatives = negatives;
	this->falseDetections = {};
	cout << "\nInitializing ViolaJones AdaBoost with " << iterations << " iterations" << endl;
	cout << "Training size: " << (positives.size() + negatives.size()) << endl;
	cout << "  -Positive samples: " << positives.size() << endl;
	cout << "  -Negative samples: " << negatives.size() << endl;
	cout << "  -Max iterations: " << iterations << "\n" << endl;

	features = {};
	features.reserve(positives.size() + negatives.size());
	features.insert(features.end(), positives.begin(), positives.end());
	features.insert(features.end(), negatives.begin(), negatives.end());

	for (int i = 0; i < features.size(); ++i) {
		/*	Initialize weights */
		if (features[i].getLabel() == 1) {
			features[i].setWeight((double) 1 / (2 * positives.size()));
		} else {
			features[i].setWeight((double) 1 / (2 * negatives.size()));
		}
	}
	initializeWeights();
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

void ViolaJones::initializeWeights(){
	for(int i = 0; i < positives.size(); ++i){
		positives[i].setWeight((double) 1 / (2 * positives.size()));
	}
	for(int i = 0; i < negatives.size(); ++i){
		negatives[i].setWeight((double) 1 / (2 * negatives.size()));
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
	cout << "Training Cascade Classifier" << endl;

	//TODO will be two attributes
	double targetFPR = 0.3;
	double minFPR = 0.65;
	double minDR = 0.7;

	double FPR = 1.0;
	double FPRold = FPR;
	double DR = 1.0;
	double DRold = DR;

	vector<Data> negativeSamples (negatives);
	vector<Data> positiveSamples (positives);

	int i = 0;
	int n = 0;
	while(FPR > targetFPR && i < maxStages){
		i++;
		n = 0;
		FPR = FPRold;
		DR = DRold;
		Stage* stage = new Stage(i - 1);
		classifier.addStage(stage);
		while(FPR > minFPR * FPRold){
			n++;
			this->iterations = n;
			initializeWeights();

			//Rearrange features
			features.clear();
			features.reserve(positiveSamples.size() + negativeSamples.size());
			features.insert(features.end(), positiveSamples.begin(), positiveSamples.end());
			features.insert(features.end(), negativeSamples.begin(), negativeSamples.end());

			cout << "  -Training size: " << features.size() << endl;

			//Train the current classifier
			StrongClassifier strongClassifier = AdaBoost::train();
			stage->setClassifiers(strongClassifier.getClassifiers());

		    //Evaluate current cascaded classifier on validation set to determine F(i) & D(i)
			pair<double, double> rates = computeRates(features);
			FPR = rates.first;
			DR = rates.second;
			stage->setFpr(FPR);
			stage->setDetectionRate(DR);

			//until the current cascaded classifier has a detection rate of at least d x D(i-1) (this also affects F(i))
			while(DR < minDR * DRold ){
				//decrease threshold for the ith classifier
				stage->decreaseThreshold(0.1);
				pair<double, double> rates = computeRates(features);
				FPR = rates.first;
				DR = rates.second;
				stage->setFpr(FPR);
				stage->setDetectionRate(DR);
			}
		}

		//N = ∅
		negativeSamples.clear();

		if(FPR > targetFPR){
			//if F(i) > Ftarget then
			//evaluate the current cascaded detector on the set of non-face images
			//and put any false detections into the set N.
			negativeSamples = falseDetections;
		}

		stage->printInfo();
		FPRold = FPR;
	}

}

pair<double, double> ViolaJones::computeRates(vector<Data> features){
	pair<double, double> output;
	falseDetections.clear();
	vector<int> predictions = classifier.predict(features);
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	for(int i = 0; i < predictions.size(); ++i){
		if(predictions[i] == 1 && features[i].getLabel() == -1){
			fp++;
			falseDetections.push_back(features[i]);
		} else if(predictions[i] == -1 && features[i].getLabel() == -1){
			tn++;
		} else if(predictions[i] == -1 && features[i].getLabel() == 1){
			fn++;
		} else if(predictions[i] == 1 && features[i].getLabel() == 1){
			tp++;
		}
	}

	//cout << "FP: " << fp << ", TN: " << tn << ", FN: " << fn << ", TP: " << tp << endl;
	output.first = (double) fp / (fp + tn);
	output.second = (double) tp / (tp + fn);
	//cout << "FPR: " << output.first << ", Detection Rate: " << output.second << endl;
	return output;
}


ViolaJones::~ViolaJones(){}
