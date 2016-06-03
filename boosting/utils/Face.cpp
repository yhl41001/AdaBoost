/*
 * Face.cpp
 *
 *  Created on: 27/mag/2016
 *      Author: lorenzocioni
 */

#include "Face.h"

Face::Face(Rect rect): rect(rect), score(0.){}

const Rect& Face::getRect() const {
	return rect;
}

void Face::setRect(const Rect& rect) {
	this->rect = rect;
}

double Face::getScore() const {
	return score;
}

void Face::setScore(double score) {
	this->score = score;
}

Face::~Face() {
}
