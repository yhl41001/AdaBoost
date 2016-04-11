/*
 * ViolaJones.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_VIOLAJONES_HPP_
#define BOOSTING_VIOLAJONES_HPP_

#include <cmath>
#include "AdaBoost.hpp"

class ViolaJones: public AdaBoost {
protected:
	void normalizeWeights();
	void updateWeights(WeakClassifier* weakClassifier);
public:
	ViolaJones(vector<Data> data, vector<double> weights, int iterations);
	~ViolaJones();
};



#endif /* BOOSTING_VIOLAJONES_HPP_ */
