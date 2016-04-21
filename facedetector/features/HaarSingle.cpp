/*
 * HaarSingle.cpp
 *
 *  Created on: 21/apr/2016
 *      Author: lorenzocioni
 */

#include "HaarSingle.h"

HaarSingle::HaarSingle(){
	this->dimension = 0;
	this->whites = {};
	this->blacks = {};
}

HaarSingle::HaarSingle(int dimension){
	this->dimension = dimension;
	this->whites = {};
	this->blacks = {};
}

HaarSingle::HaarSingle(int dimension, vector<Rect> whites, vector<Rect> blacks){
	this->dimension = dimension;
	this->whites = whites;
	this->blacks = blacks;
}

void HaarSingle::addWhite(Rect w){
	whites.push_back(w);
}

void HaarSingle::addBlack(Rect b){
	blacks.push_back(b);
}

void HaarSingle::toString(){
	cout << "haar: " << dimension;
	cout <<"whites" << whites.size();
	cout << "blacks" << blacks.size();
}

const vector<Rect>& HaarSingle::getBlacks() const {
	return blacks;
}

void HaarSingle::setBlacks(const vector<Rect>& blacks) {
	this->blacks = blacks;
}

int HaarSingle::getDimension() const {
	return dimension;
}

void HaarSingle::setDimension(int dimension) {
	this->dimension = dimension;
}

const vector<Rect>& HaarSingle::getWhites() const {
	return whites;
}

void HaarSingle::setWhites(const vector<Rect>& whites) {
	this->whites = whites;
}

double HaarSingle::evaluate(Mat intImg){
	double white = 0;
	double black = 0;
	for(int w = 0; w < whites.size(); ++w){
		white += IntegralImage::computeArea(intImg, whites[w]);
	}
	for(int b = 0; b < whites.size(); ++b){
		black += IntegralImage::computeArea(intImg, blacks[b]);
	}
	return white - black;
}

HaarSingle::~HaarSingle(){}
