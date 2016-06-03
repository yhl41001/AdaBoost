/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.h"

ViolaJones::ViolaJones(): AdaBoost(){
	this->maxStages = 0;
	this->negativeSetLayer = 0;
}

ViolaJones::ViolaJones(string trainedPath): AdaBoost(){
	this->maxStages = 0;
	this->iterations = 0;
	this->classifier = *(new CascadeClassifier());
	this->negativeSetLayer = 0;
	loadTrainedData(trainedPath);
}

ViolaJones::ViolaJones(vector<Data*> positives, vector<Data*> negatives, int maxStages):
	AdaBoost(){
	this->iterations = 0;
	this->maxStages = maxStages;
	this->classifier = *(new CascadeClassifier());
	this->positives = positives;
	this->negatives = negatives;
	cout << "\nInitializing ViolaJones AdaBoost with " << iterations << " iterations" << endl;
	cout << "Training size: " << (positives.size() + negatives.size()) << endl;
	cout << "  -Positive samples: " << positives.size() << endl;
	cout << "  -Negative samples: " << negatives.size() << endl;
	this->features = {};
	this->negativeSetLayer = positives.size();
}

double ViolaJones::updateAlpha(double error){
	if(error < 0.0001){
		return 10000;
	}
	return  log((1 - error) / error);
}

double ViolaJones::updateBeta(double error){
	return error / (1 - error);
}

void ViolaJones::normalizeWeights(){
	double norm = 0;
	for (int i = 0; i < features.size(); ++i) {
		norm += features[i]->getWeight();
	}
	for (int i = 0; i < features.size(); ++i) {
		features[i]->setWeight((double) features[i]->getWeight() / norm);
	}
}

void ViolaJones::initializeWeights(){
	for(int i = 0; i < positives.size(); ++i){
		positives[i]->setWeight((double) 1 / (2 * positives.size()));
	}
	for(int i = 0; i < negatives.size(); ++i){
		negatives[i]->setWeight((double) 1 / (2 * negatives.size()));
	}
}

void ViolaJones::updateWeights(WeakClassifier* weakClassifier){
	for(int i = 0; i < features.size(); ++i){
		int e = (features[i]->getLabel()
				* weakClassifier->predict(this->features[i]) > 0) ? 0 : 1;
		double num = features[i]->getWeight() * (pow(weakClassifier->getBeta(), (double) (1 - e)));
		features[i]->setWeight(num);
	}
}

void ViolaJones::train(){
	cout << "Training Cascade Classifier" << endl;

	double f = 0.5;
	double d = 0.95;
	double Ftarget = 0.00001;
	double* F = new double[maxStages + 1];
	double* D = new double[maxStages + 1];
	double fpr, dr;
	pair<double, double> rates;
	vector<WeakClassifier> classifiers;

	vector<Data*> validationSet;
	validationSet.reserve(positives.size() + negatives.size());
	validationSet.insert(validationSet.end(), positives.begin(), positives.end());
	validationSet.insert(validationSet.end(), negatives.begin(), negatives.end());

	F[0] = 1.;
	D[0] = 1.;

	int i = 0;
	int n;

	while(F[i] > Ftarget && i < maxStages){
		if(negatives.size() == 0){
			cout << "All training negative samples classified correctly. Could not achieve validation target FPR for this stage." << endl;
			break;
		}

		i++;
		n = 0;

		if(i > 0){
			F[i] = F[i - 1];
		}

		classifiers.clear();
		initializeWeights();

		//Rearrange features
		features.clear();
		features.reserve(positives.size() + negatives.size());
		features.insert(features.end(), positives.begin(), positives.end());
		features.insert(features.end(), negatives.begin(), negatives.end());

		cout << "\n*** Stage n. " << i << " ***\n" << endl;
		cout << "  -Training size: " << features.size() << endl;
		Stage* stage = new Stage(i);
		classifier.addStage(stage);

		fpr = i > 0 ? F[i - 1] : 1;
		while(F[i] > f * fpr){
			n++;
			this->iterations = n;
			normalizeWeights();

			//Train the current classifier
			StrongClassifier* strongClassifier = AdaBoost::train(classifiers);
			stage->setClassifiers(strongClassifier->getClassifiers());
			classifiers = strongClassifier->getClassifiers();

			rates = computeRates(validationSet);
			F[i] = rates.first;
			D[i] = rates.second;

			//Optimizing stage threshold
			//Evaluate current cascaded classifier on validation set to determine fpr & dr
			dr = i > 0 ? D[i - 1] : 1;

			while(D[i] < d * dr){
				stage->decreaseThreshold();
				rates = computeRates(validationSet);
				F[i] = rates.first;
				D[i] = rates.second;
			}

			stage->setFpr(F[i]);
			stage->setDetectionRate(D[i]);
		}

		//N = ∅
		negatives.clear();

		if(F[i] > Ftarget){
			//if F(i) > Ftarget then
			//evaluate the current cascaded detector on the set of non-face images
			//and put any false detections into the set N.
			//negatives = falseDetections;
			generateNegativeSet();
		}
		stage->printInfo();
	}
	store();
}

pair<double, double> ViolaJones::computeRates(vector<Data*> validationSet){
	falseDetections.clear();
	pair<double, double> output;
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	int prediction;
	for(int i = 0; i < validationSet.size(); ++i){
		prediction = classifier.predict(validationSet[i]->getFeatures());
		if(prediction == 1 && validationSet[i]->getLabel() == -1){
			fp++;
			falseDetections.push_back(validationSet[i]);
		} else if(prediction == -1 && validationSet[i]->getLabel() == -1){
			tn++;
		} else if(prediction == -1 && validationSet[i]->getLabel() == 1){
			fn++;
		} else if(prediction == 1 && validationSet[i]->getLabel() == 1){
			tp++;
		}
	}
	output.first = (double) fp / (fp + tn);
	output.second = (double) tp / (tp + fn);
	cout << "FPR: " << output.first << ", DR: " << output.second;
	cout << " (FP: " << fp << " FN: " << fn << " TN: " << tn << " TP: " << tp << ")" << endl;
	return output;
}


void ViolaJones::generateNegativeSet(){
	cout << "\nGenerating negative set for layer: max " << negativeSetLayer << endl;
	string negativePath = "/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/dataset/negatives";
	vector<string> negativeImages = Utils::open(negativePath);
	int count = 0;
	for(int k = 0; k < negativeImages.size() && count < negativeSetLayer; ++k){
		Mat img = imread(negativePath + "/" + negativeImages[k]);
		Mat dest;
		if(img.rows > 0 && img.cols > 0){
			for(int f = -2; f < 2; ++f){
				if(f > -2){
					flip(img, dest, f);
				} else {
					dest = img;
				}
				Mat intImg = IntegralImage::computeIntegralImage(dest);
				if(classifier.predict(intImg) == 1){
					vector<double> features = HaarFeatures::extractFeatures(intImg, 24, 0, 0);
					negatives.push_back(new Data(features, -1));
					count++;
					cout << "\rAdded " << count << " images to the negative set" << flush;
				}
			}
		}
	}
}

int ViolaJones::predict(Mat img){
	return classifier.predict(img);
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
				<< "," << stage->getThreshold() << "\n";

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
			//Haar haar;
			HaarFeatures::getFeature(24, wc);
//			selectedFeatures.push_back(haar);
			stage->addClassifier(wc);
		}
	}

	readFile.close();
	cout << "Trained data loaded correctly" << endl;
}



vector<Face> ViolaJones::mergeDetections(vector<Face> &detections){
	vector<Face> output;
	vector<int> indexes;
	double ci, cj;
	double distance;
	bool insert;
	double score;
	double th = 6;
	int padding = 4;
	int size;
	for(unsigned int j = 0; j < detections.size(); ++j){
		size = output.size();
		insert = false;
		score = 1;
		if(size == 0){
			insert = true;
		} else {
			cj = (detections[j].getRect().x + detections[j].getRect().width) / 2;
			indexes.clear();
			for(unsigned int i = 0; i < output.size(); ++i){
				ci = (output[i].getRect().x + output[i].getRect().width) / 2;
				distance = sqrt(2 * pow(ci - cj, 2));
				if(distance < th){
					output.erase(output.begin() + i);
					score += output[i].getScore();
					i--;

					//indexes.push_back(i);
				}
			}
			if(indexes.size() > 0){
				cout << "indexes: "<< indexes.size() << endl;
				for(unsigned int k = 0; k < indexes.size(); ++k){
					if(output[indexes[k]].getRect().width < detections[j].getRect().width){
						output.erase(output.begin() + indexes[k]);
						score += output[indexes[k]].getScore();
						insert = true;
					}
				}
			} else {
				insert = true;
			}
		}
		if(insert){
			detections[j].setScore(score);
			output.push_back(detections[j]);
		}
	}

	cout << "size: " << output.size() << endl;

	cout << "remained " << output.size() << endl;
	indexes.clear();
	for(int h = 0; h < output.size(); ++h){
		cout << "index: " << h << " - " << output[h].getRect() << " value: " << output[h].getScore() <<  endl;
		}

	output.erase(remove_if(output.begin(), output.end(), [output](Face& face){
		for(unsigned int i = 0; i < output.size(); ++i){
			double intersection = (output[i].getRect() & face.getRect()).area();
			if(intersection == face.getRect().area() && intersection != output[i].getRect().area()){
				 return true;
			}
		}
		return false;
	}), output.end());

	cout << "size: " << output.size() << endl;

	double norm;
	for_each(output.begin(), output.end(), [&norm] (const Face& face) {
		norm += face.getScore();
	});
	double max = 0;
	for_each(output.begin(), output.end(), [&norm, &max] (Face& face) {
		double s = face.getScore()/norm;
		face.setScore(s);
		if(s > max) max = s;
	});
	th = max * 60 / 100;
	for(unsigned int i = 0; i < output.size(); ++i){
		if(output[i].getScore() < th){
			//output.erase(output.begin() + i);
			//i--;
		} else {
			Rect r = output[i].getRect();
			r.x -= padding;
			r.y -= padding;
			r.width += 2*padding;
			r.height += 2*padding;
			output[i].setRect(r);
		}
	}

	cout << "end " << endl;
	for(int h = 0; h < output.size(); ++h){
			cout << "index: " << h << " - " << output[h].getRect() << " value: " << output[h].getScore() <<  endl;
			}

	return output;
}

ViolaJones::~ViolaJones(){}
