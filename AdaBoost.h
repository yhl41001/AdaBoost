/*
 * AdaBoost.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <string>
#include <vector>

using namespace std;

class AdaBoost {

private:
	int iterations;
	vector<double> weights;

public:
	AdaBoost();
	int getIterations() const;
	void setIterations(int iterations);
	~AdaBoost();
};


#endif /* ADABOOST_H_ */
