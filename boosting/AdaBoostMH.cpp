/*
 * AdaBoostMH.cpp
 *
 *  Created on: 18/mag/2016
 *      Author: lorenzocioni
 */

#include "AdaBoostMH.h"

AdaBoostMH::AdaBoostMH(vector<Data*> data, int iterations, int classes):
	iterations(iterations),
	classifier(new MultiClassClassifier()),
	features(data){
	int size = features.size();
	this->classes = classes;
	cout << "Initializing AdaBoost.MH with " << iterations << " iterations" << endl;
	cout << "Training size: " << size << "\n" << endl;
}

/**
 * Updates features weights according to their errors
 */
void AdaBoostMH::updateWeights(MultiWeakClassifier* weakClassifier){
	double norm = 0;
	for(int i = 0; i < features.size(); ++i){
		double num = (features[i]->getWeight() * exp(-weakClassifier->getAlpha()
				* features[i]->getLabel() * weakClassifier->predict(features[i])));
		norm += num;
		features[i]->setWeight(num);
	}
	for(int i = 0; i < features.size(); ++i){
		//Normalize such that wt+1 is a prob. distribution
		features[i]->setWeight((double) features[i]->getWeight()/norm);
	}
}

MultiClassClassifier* AdaBoostMH::train(){
	int size = features.size();

	vector<vector<Data*>> data(size, vector<Data*>(classes));
	for(unsigned int i = 0; i < size ; ++i){
		for(unsigned int k = 0; k < classes; ++k){
			if(features[i]->getLabel() == k){
				data[i][k] = new Data(features[i]->getFeatures(), 1, k);
			} else {
				data[i][k] = new Data(features[i]->getFeatures(), -1, k);
			}
		}
	}
	this->data = data;

	//Initialize weights
	for (int n = 0; n < data.size(); ++n) {
		for(int d = 0; d < data[n].size(); ++d){
			data[n][d]->setWeight((double) 1 / (classes * features.size()));
		}
	}

	cout << "Training AdaBoost.MH with " << iterations << " iterations" << endl;
	auto t_start = chrono::high_resolution_clock::now();
	vector<MultiWeakClassifier> classifiers;
	//Iterate for the specified iterations
	for (unsigned int i = 0; i < iterations; ++i) {
		cout << "Iteration: " << (i + 1) << endl;;
		MultiWeakClassifier* weakClassifier = trainWeakClassifier();
		double error = weakClassifier->getError();
		if(error < 0.5){
			double alpha = updateAlpha(error);
			double beta = updateBeta(error);
			weakClassifier->setAlpha(alpha);
			weakClassifier->setBeta(beta);
			updateWeights(weakClassifier);
			weakClassifier->printInfo();
			classifiers.push_back(*weakClassifier);
			//If error is 0, classification is perfect (linearly separable data)
			if(error == 0){
				break;
			}
		} else {
			cout << "Error: weak classifier with error > 0.5." << endl;
		}
	}
	//showFeatures();
	//Create strong classifier

    auto t_end = high_resolution_clock::now();
    cout << "Time: " << (duration<double, milli>(t_end - t_start).count())/1000 << " s" << endl;

	return new MultiClassClassifier();
}

/***
 * Train weak classifier on training data choosing the one minimizing the error
 */
MultiWeakClassifier* AdaBoostMH::trainWeakClassifier(){
	MultiWeakClassifier* bestWeakClass = new MultiWeakClassifier(classes);
	double sumEdges;
	vector<double> edges(classes);

	for(unsigned int l = 0; l < classes; ++l){
		sumEdges = 0;
		for(unsigned int i = 0; i < data.size(); ++i){
			sumEdges += data[i][l]->getWeight() * data[i][l]->getClas();
		}
		edges[l] = sumEdges;
	}

	int dimensions = data[0][0]->getFeatures().size();
	double th;
	double coeff;
	vector<double> alphas(dimensions);
	//Iterate through dimensions
	for (unsigned int j = 0; j < dimensions; ++j) {
		//Sorts vector of features according to the j-th dimension
		sort(data.begin(), data.end(),
				[j](vector<Data*> const &a, vector<Data*> const &b) {return a[0]->getFeatures()[j] < b[0]->getFeatures()[j];});
		vector<int> signs;
		findBestThreshold(edges, j, signs, th, coeff);
		alphas[j] = 0.5 * log((1 + coeff)/(1 - coeff));
	}

	//Continua riga 7 pagina 7 kegl14.pdf


	return bestWeakClass;
}

void AdaBoostMH::findBestThreshold(vector<double> edges, int dim, vector<int> &signs, double &th, double &coeff){
	vector<double> bestEdges(edges);
	vector<double> initialEdges(edges);

	double s1, s2;

	double initialSum, bestSum;
	for(unsigned int i = 0; i < data.size() - 1; ++i){
		for(unsigned int l = 0; l < classes; ++l){
			initialEdges[l] = initialEdges[l] - 2 * data[i][l]->getWeight() * data[i][l]->getClas();
		}

		s1 = data[i][0]->getFeatures()[dim];
		s2 = data[i + 1][0]->getFeatures()[dim];
		if(s1 != s2){
			initialSum = 0;
			bestSum = 0;
			for(unsigned int k = 0; k < classes; ++k){
				initialSum += abs(initialEdges[k]);
				bestSum += abs(bestEdges[k]);
			}
			if(initialSum > bestSum){
				copy(initialEdges.begin(), initialEdges.end(), bestEdges.begin());
				th = (double) (s1 + s2) / 2;
			}
		}
	}

	signs.reserve(classes);
	for(unsigned int l = 0; l < classes; ++l){
		if(initialEdges[l] >= 0){
			signs[l] = 1;
		} else {
			signs[l] = -1;
		}
	}

	coeff = 0;
	for(unsigned int k = 0; k < classes; ++k){
		coeff += abs(bestEdges[k]);
	}
	if(equal(initialEdges.begin(), initialEdges.end(), bestEdges.begin() )){
		th = numeric_limits<double>::min();
	}
}

double AdaBoostMH::updateAlpha(double error){
	return  0.5 * log((1 - error) / error);
}

double AdaBoostMH::updateBeta(double error){
	return error / (1 - error);
}

AdaBoostMH::~AdaBoostMH(){

}
