/*
 * ViolaJones.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_VIOLAJONES_HPP_
#define BOOSTING_VIOLAJONES_HPP_

#include "AdaBoost.h"

class ViolaJones: public AdaBoost {
private:
	void normalizeWeights();
	void updateWeights(WeakClassifier* weakClassifier);
public:
	ViolaJones(vector<Data> data, vector<double> weights, int iterations);
};



#endif /* BOOSTING_VIOLAJONES_HPP_ */
