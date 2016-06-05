/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.h"

ViolaJones::ViolaJones(): AdaBoost(){
	this->maxStages = 0;
	this->negativesPerLayer = 0;
	this->detectionWindowSize = 24;
	this->validationPath = "";
	this->numPositives = 0;
	this->numNegatives = 0;
}

ViolaJones::ViolaJones(string trainedPath): AdaBoost(){
	this->maxStages = 0;
	this->iterations = 0;
	this->classifier = *(new CascadeClassifier());
	this->negativesPerLayer = 0;
	this->validationPath = "";
	this->numPositives = 0;
	this->numNegatives = 0;
	loadTrainedData(trainedPath);
}

ViolaJones::ViolaJones(string positivePath, string negativePath, int maxStages, int numPositives, int numNegatives, int detectionWindowSize, int negativesPerLayer):
	AdaBoost(){
	this->iterations = 0;
	this->maxStages = maxStages;
	this->classifier = *(new CascadeClassifier());
	this->positivePath = positivePath;
	this->negativePath = negativePath;
	this->validationPath = "";
	this->detectionWindowSize = detectionWindowSize;
	this->features = {};
	this->numPositives = numPositives;
	this->numNegatives = numNegatives;
	if(negativesPerLayer == 0){
		this->negativesPerLayer = numPositives;
	} else {
		this->negativesPerLayer = negativesPerLayer;
	}
}

double ViolaJones::updateAlpha(double error){
	if(error < 0.0001){
		return 1000;
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

void ViolaJones::setValidationPath(const string& validationPath) {
	this->validationPath = validationPath;
}

void ViolaJones::extractFeatures(){
	//Loading training positive images
	double percent;
	int count = 0;
	Mat img, intImg;

	cout << "Extracting image features" << endl;
	auto t_start = chrono::high_resolution_clock::now();

	vector<string> positiveImages = Utils::open(positivePath);
	vector<string> negativeImages = Utils::open(negativePath);

	int totalExamples = numPositives + numNegatives;
	cout << "Training size: " << totalExamples << endl;
	if (numPositives > positiveImages.size())
		numPositives = positiveImages.size();
	cout << "  -Positive samples: " << numPositives << endl;
	if (numNegatives > negativeImages.size())
		numNegatives = negativeImages.size();
	cout << "  -Negative samples: " << numNegatives << endl;

	if(validationPath != ""){
		vector<string> validationImages = Utils::open(validationPath);
		cout << "  -Validation set size: " << validationImages.size() << endl;
		totalExamples += validationImages.size();
		for (int k = 0; k < validationImages.size(); ++k) {
			img = imread(validationPath + validationImages[k]);
			if (img.rows != 0 && img.cols != 0) {
				Mat dest;
				resize(img, dest, Size(detectionWindowSize, detectionWindowSize));
				intImg = IntegralImage::computeIntegralImage(dest);
				vector<double> features = HaarFeatures::extractFeatures(intImg,
						detectionWindowSize, 0, 0);
				validation.push_back(new Data(features, -1));
				percent = (double) count * 100 / totalExamples;
				count++;
				cout << "\rEvaluated: " << count << "/" << totalExamples << " images" << flush;
			}
		}
	}


	for (int k = 0; k < numPositives; ++k) {
		img = imread(positivePath + positiveImages[k]);
		if (img.rows != 0 && img.cols != 0) {
			Mat dest;
			resize(img, dest, Size(detectionWindowSize, detectionWindowSize));
			intImg = IntegralImage::computeIntegralImage(dest);
			vector<double> features = HaarFeatures::extractFeatures(intImg,
					detectionWindowSize, 0, 0);
			positives.push_back(new Data(features, 1));
			percent = (double) count * 100 / totalExamples;
			count++;
			cout << "\rEvaluated: " << count << "/" << totalExamples << " images" << flush;

		}
	}


	for (int k = 0; k < numNegatives; ++k) {
		Mat img = imread(negativePath + negativeImages[k]);
		if (img.rows != 0 && img.cols != 0) {
			Mat dest;
			resize(img, dest, Size(detectionWindowSize, detectionWindowSize));
			intImg = IntegralImage::computeIntegralImage(dest);
			vector<double> features = HaarFeatures::extractFeatures(intImg,
					detectionWindowSize, 0, 0);
			negatives.push_back(new Data(features, -1));
			percent = (double) count * 100 / totalExamples;
			count++;
			cout << "\rEvaluated: " << count << "/" << totalExamples << " images" << flush;
		}
	}
	cout << "\nExtracted features in ";
	auto t_end = chrono::high_resolution_clock::now();
	cout << std::fixed
			<< (chrono::duration<double, milli>(t_end - t_start).count()) / 1000
			<< " s\n" << endl;
}

void ViolaJones::train(){
	cout << "Traing ViolaJones face detector\n" << endl;
	extractFeatures();

	double f = 0.5;
	double d = 0.98;
	double Ftarget = 0.00001;
	double* F = new double[maxStages + 1];
	double* D = new double[maxStages + 1];
	double fpr, dr;
	vector<WeakClassifier> classifiers;

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

		if(validation.size() == 0){
			validation.reserve(negatives.size());
			validation.insert(validation.end(), negatives.begin(), negatives.end());
		}

		cout << "\n*** Stage n. " << i << " ***\n" << endl;
		cout << "  -Training size: " << features.size() << endl;
		Stage* stage = new Stage(i);
		classifier.addStage(stage);

		fpr = i > 0 ? F[i - 1] : 1;
		cout << "  -Target FPR: " << (f * fpr) << endl;
		cout << "  -Target DR: " << (d * dr) << "\n" << endl;

		while(F[i] > f * fpr){
			n++;
			this->iterations = n;
			normalizeWeights();

			//Train the current classifier
			StrongClassifier* strongClassifier = AdaBoost::train(classifiers);
			stage->setClassifiers(strongClassifier->getClassifiers());
			classifiers = strongClassifier->getClassifiers();

			D[i] = evaluateDR(positives);

			//Optimizing stage threshold
			//Evaluate current cascaded classifier on validation set to determine fpr & dr
			dr = i > 0 ? D[i - 1] : 1;

			while(D[i] < d * dr){
				stage->decreaseThreshold();
				D[i] = evaluateDR(positives);
			}

			F[i] = evaluateFPR(validation);
			stage->setFpr(F[i]);
			stage->setDetectionRate(D[i]);
		}

		//N = ∅
		negatives.clear();

		if(F[i] > Ftarget){
			//if F(i) > Ftarget then
			//evaluate the current cascaded detector on the set of non-face images
			//and put any false detections into the set N.
			generateNegativeSet();
		}
		stage->printInfo();
	}
	store();
}

double ViolaJones::evaluateFPR(vector<Data*> validationSet){
	cout << "Evaluate FPR on validation set:" << endl;
	int fp = 0;
	int tn = 0;
	int prediction;
	for(int i = 0; i < validationSet.size(); ++i){
		prediction = classifier.predict(validationSet[i]->getFeatures());
		if(prediction == 1 && validationSet[i]->getLabel() == -1){
			fp++;
		} else if(prediction == -1 && validationSet[i]->getLabel() == -1){
			tn++;
		}
	}
	double fpr = (double) fp / (fp + tn);
	cout << "FPR: " << fpr;
	cout << " (FP: " << fp << " TN: " << tn << ")\n" << endl;
	return fpr;
}

double ViolaJones::evaluateDR(vector<Data*> validationSet){
	int tp = 0;
	int fn = 0;
	int prediction;
	for(int i = 0; i < validationSet.size(); ++i){
		prediction = classifier.predict(validationSet[i]->getFeatures());
		if(prediction == -1 && validationSet[i]->getLabel() == 1){
			fn++;
		} else if(prediction == 1 && validationSet[i]->getLabel() == 1){
			tp++;
		}
	}
	double dr = (double) tp / (tp + fn);
	cout << "DR: " << dr;
	cout << " (TP: " << tp << " FN: " << fn << ")" << endl;
	return dr;
}


void ViolaJones::generateNegativeSet(){
	cout << "\nGenerating negative set for layer: max " << negativesPerLayer << endl;
	vector<string> negativeImages = Utils::open(negativePath);
	int count = 0;
	int evaluated = 0;
	for(int k = 0; k < negativeImages.size() && count < negativesPerLayer; ++k){
		Mat img = imread(negativePath + negativeImages[k]);
		Mat dest;
		if(img.rows > 0 && img.cols > 0){
			for(int f = -2; f < 2; ++f){
				if(f > -2){
					flip(img, dest, f);
				} else {
					dest = img;
				}
				evaluated++;
				Mat intImg = IntegralImage::computeIntegralImage(dest);
				vector<double> features = HaarFeatures::extractFeatures(intImg, 24, 0, 0);
				if(classifier.predict(features) == 1){
					negatives.push_back(new Data(features, -1));
					count++;
					cout << "\rAdded " << count << " (" << evaluated << " tested) images to the negative set" << flush;
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
			HaarFeatures::getFeature(24, wc);
			stage->addClassifier(wc);
		}
	}

	readFile.close();
	cout << "Trained data loaded correctly" << endl;
}

vector<Face> ViolaJones::mergeDetections(vector<Face> detections, int padding, double th){
	vector<Face> output, cluster;
	double score;
	Rect a, b;

	sort(detections.begin(), detections.end(),
			[](Face const &a, Face const &b) {return a.getRect().area() > b.getRect().area();});

	for (unsigned int i = 0; i < detections.size(); ++i) {
		if (!detections[i].isEvaluated()) {
			cluster.clear();
			cluster.push_back(detections[i]);
			detections[i].setEvaluated(true);
			a = detections[i].getRect();

			for (unsigned int j = 0; j < detections.size(); ++j) {
				if (i != j && !detections[j].isEvaluated()) {
					b = detections[j].getRect();
					score = (double) (a & b).area() / (a | b).area();
					if (score > th) {
						detections[j].setEvaluated(true);
						cluster.push_back(detections[j]);
					}
				}
			}

			if(cluster.size() > 3){
				Rect result(0, 0, 0, 0);

				for (unsigned int k = 0; k < cluster.size(); ++k) {
					result.x += cluster[k].getRect().x;
					result.y += cluster[k].getRect().y;
					result.width += cluster[k].getRect().width;
					result.height += cluster[k].getRect().height;
				}

				result.x = result.x / cluster.size() - padding;
				result.y = result.y / cluster.size() - padding;
				result.width = result.width / cluster.size() + 2*padding;
				result.height = result.height / cluster.size() + 2*padding;

				output.push_back(Face(result, (double) cluster.size()));
			}

		}
	}
	return output;
}

const string& ViolaJones::getValidationPath() const {
	return validationPath;
}

const CascadeClassifier& ViolaJones::getClassifier() const {
	return classifier;
}

void ViolaJones::setClassifier(const CascadeClassifier& classifier) {
	this->classifier = classifier;
}

ViolaJones::~ViolaJones() {
}
