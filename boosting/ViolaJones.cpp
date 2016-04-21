/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.h"

ViolaJones::ViolaJones(): AdaBoost(){
	this->maxStages = 0;
	this->selectedFeatures = {};
	this->falseDetections = {};
}

ViolaJones::ViolaJones(string trainedPath): AdaBoost(){
	this->maxStages = 0;
	this->iterations = 0;
	this->falseDetections = {};
	this->selectedFeatures = {};
	this->classifier = *(new CascadeClassifier());
	loadTrainedData(trainedPath);
}

ViolaJones::ViolaJones(vector<Data> positives, vector<Data> negatives, int maxStages):
	AdaBoost(){
	this->iterations = 0;
	this->maxStages = maxStages;
	this->classifier = *(new CascadeClassifier());
	this->positives = positives;
	this->negatives = negatives;
	this->falseDetections = {};
	this->selectedFeatures = {};
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
	double DRtmp;

	vector<Data> negativeSamples (negatives);
	vector<Data> positiveSamples (positives);

	int i = 0;
	int n = 0;
	while(FPR > targetFPR && i < maxStages){
		if(negativeSamples.size() == 0){
			cout << "All training negative samples classified correctly. Could not achieve validation target FPR for this stage." << endl;
			break;
		}
		i++;
		n = 0;
		FPR = FPRold;
		DR = DRold;
		Stage* stage = new Stage(i - 1);
		cout << "\n*** Stage n. " << i - 1 << " ***\n" << endl;
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
			pair<double, double> rates = computeRates();
			FPR = rates.first;
			DR = rates.second;
			DRtmp = 1.1;
			stage->setFpr(FPR);
			stage->setDetectionRate(DR);

			//until the current cascaded classifier has a detection rate of at least d x D(i-1) (this also affects F(i))
			while(DR < minDR * DRold && DR != DRtmp){
				//decrease threshold for the ith classifier
				stage->decreaseThreshold(0.1);
				DRtmp = DR;
				rates = computeRates();
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
	store();
}

pair<double, double> ViolaJones::computeRates(){
	pair<double, double> output;
	falseDetections.clear();
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	int prediction;
	for(int i = 0; i < features.size(); ++i){
		prediction = classifier.predict(features[i]);
		if(prediction == 1 && features[i].getLabel() == -1){
			fp++;
			falseDetections.push_back(features[i]);
		} else if(prediction == -1 && features[i].getLabel() == -1){
			tn++;
		} else if(prediction == -1 && features[i].getLabel() == 1){
			fn++;
		} else if(prediction == 1 && features[i].getLabel() == 1){
			tp++;
		}
	}
	output.first = (double) fp / (fp + tn);
	output.second = (double) tp / (tp + fn);

	return output;
}

int ViolaJones::predict(Data x){
	return classifier.predict(x);
}

void ViolaJones::store(){
	cout << "\nStoring trained face detector" << endl;
	ofstream output, data;
	output.open ("trainedInfo.txt");
	data.open ("trainedData.txt");

	WeakClassifier wc;

    for(unsigned int i = 0; i < classifier.getStages().size(); ++i){
    	Stage* stage = classifier.getStages()[i];

    	//Outputs info
    	output << "Stage " << i << "\n\n";
    	output << "FPR: " << stage->getFpr() << "\n";
    	output << "DR: " << stage->getDetectionRate() << "\n";
    	output << "Threshold: " << stage->getThreshold() << "\n";
    	output << "Classifiers:\n" << endl;
    	//Output data
		data << "s:" << stage->getFpr() << "," << stage->getDetectionRate()
				<< "," << stage->getDetectionRate() << "\n";

    	for(unsigned int j = 0; j < stage->getClassifiers().size(); ++j){
    		wc = stage->getClassifiers()[j];
    		output << "WeakClassifier " << j << "\n";
    		output << "Error: " << wc.getError() << "\n";
    		output << "Dimension: " << wc.getDimension() << "\n";
    		output << "Threshold: " << wc.getThreshold() << "\n";
    		output << "Alpha: " << wc.getAlpha() << "\n";
    		output << "Beta: " << wc.getBeta() << "\n";
    		if(wc.getSign() == POSITIVE){
    			output << "Sign: POSITIVE\n";
    		} else {
    			output << "Sign: NEGATIVE\n";
    		}
    		output << "Misclassified: " << wc.getMisclassified() << "\n\n";
    		//Outputs data
			data << "c:" << wc.getError() << "," << wc.getDimension() << ","
					<< wc.getThreshold() << "," << wc.getAlpha() << ","
					<< wc.getBeta() << ",";
			if (wc.getSign() == POSITIVE) {
				data << "POSITIVE,";
			} else {
				data << "NEGATIVE,";
			}
			data << wc.getMisclassified() << "\n";
    	}

    	output << "---------------\n" << endl;
	}

    output.close();
}

void ViolaJones::loadTrainedData(string filename){
	cout << "Loading data from file: " << filename << endl;
	string line;
	string read;
	ifstream readFile(filename);
	Stage* stage;
	WeakClassifier* wc;

	while(getline(readFile,line)){
		stringstream iss(line);
		getline(iss, read, ':');
		if (read.compare("s") == 0) {
			//Found stage
			stage = new Stage(classifier.getStages().size());
			getline(iss, read, ',');
			stage->setFpr(stod(read));
			getline(iss, read, ',');
			stage->setDetectionRate(stod(read));
			getline(iss, read, ',');
			stage->setThreshold(stod(read));
			classifier.addStage(stage);
		} else if (read.compare("c") == 0) {
			//Found classifier
			wc = new WeakClassifier();
			getline(iss, read, ',');
			wc->setError(stod(read));
			getline(iss, read, ',');
			wc->setDimension(stoi(read));
			getline(iss, read, ',');
			wc->setThreshold(stod(read));
			getline(iss, read, ',');
			wc->setAlpha(stod(read));
			getline(iss, read, ',');
			wc->setBeta(stod(read));
			getline(iss, read, ',');
			if (read.compare("POSITIVE") == 0) {
				wc->setSign(POSITIVE);
			} else {
				wc->setSign(NEGATIVE);
			}
			getline(iss, read, ',');
			wc->setMisclassified(stoi(read));
			vector<Rect> whites;
			vector<Rect> blacks;
			HaarSingle haar;
			HaarFeatures::getFeature(24, wc->getDimension(), haar);
			selectedFeatures.push_back(haar);
			stage->addClassifier(wc);
		}
	}

	readFile.close();
	cout << "Trained data loaded correctly" << endl;
}

const vector<HaarSingle>& ViolaJones::getSelectedFeatures() const {
	return selectedFeatures;
}

void ViolaJones::setSelectedFeatures(
		const vector<HaarSingle>& selectedFeatures) {
	this->selectedFeatures = selectedFeatures;
}

ViolaJones::~ViolaJones(){}
