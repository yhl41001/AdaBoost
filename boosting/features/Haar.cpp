/*
 * HaarSingle.cpp
 *
 *  Created on: 21/apr/2016
 *      Author: lorenzocioni
 */

#include "Haar.h"

Haar::Haar(){
	this->dimension = 0;
	this->whites = {};
	this->blacks = {};
	this->value = 0;
}

Haar::Haar(int dimension){
	this->dimension = dimension;
	this->whites = {};
	this->blacks = {};
	this->value = 0;
}

Haar::Haar(int dimension, vector<Rect> whites, vector<Rect> blacks){
	this->dimension = dimension;
	this->whites = whites;
	this->blacks = blacks;
	this->value = 0;
}

void Haar::addWhite(Rect w){
	whites.push_back(w);
}

void Haar::addBlack(Rect b){
	blacks.push_back(b);
}

void Haar::toString(){
	cout << "haar: " << dimension;
	cout <<"whites" << whites.size();
	cout << "blacks" << blacks.size();
}

const vector<Rect>& Haar::getBlacks() const {
	return blacks;
}

void Haar::setBlacks(const vector<Rect>& blacks) {
	this->blacks = blacks;
}

int Haar::getDimension() const {
	return dimension;
}

void Haar::setDimension(int dimension) {
	this->dimension = dimension;
}

const vector<Rect>& Haar::getWhites() const {
	return whites;
}

void Haar::setWhites(const vector<Rect>& whites) {
	this->whites = whites;
}

double Haar::evaluate(Mat intImg){
	double white = 0;
	double black = 0;
	for(int w = 0; w < whites.size(); ++w){
		white += IntegralImage::computeArea(intImg, whites[w]);
	}
	for(int b = 0; b < blacks.size(); ++b){
		black += IntegralImage::computeArea(intImg, blacks[b]);
	}
	value = white - black;
	return value;
}

Haar::~Haar() {
}

double Haar::getValue() const {
	return value;
}

void Haar::setValue(double value) {
	this->value = value;
}
