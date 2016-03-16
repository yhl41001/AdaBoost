/*
 * utils.h
 *
 *  Created on: 16/mar/2016
 *      Author: lorenzocioni
 */

#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

enum example {POSITIVE, NEGATIVE};

//Sign function
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

#endif /* UTILS_UTILS_H_ */
