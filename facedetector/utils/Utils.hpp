/*
 * Utils.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_UTILS_UTILS_HPP_
#define FACEDETECTOR_UTILS_UTILS_HPP_

#include <dirent.h>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

using namespace std;

class Utils {
public:
	static vector<string> open(string path = ".") {
	    DIR*    dir;
	    dirent* pdir;
	    vector<string> files;

	    dir = opendir(path.c_str());

	    while (pdir = readdir(dir)) {
	    	if(strcmp( pdir->d_name, ".") != 0 && strcmp(pdir->d_name, "..") != 0 )
	    		files.push_back(pdir->d_name);
	    }

	    return files;
	}

	static void generateNonFacesDataset(string path, string outputDir, int number, int size){
		cout << "Generating non faces dataset from given images" << endl;
		vector<string> images = open(path);
		int counter = 0;
		int k = 0;
		int delta = 20;
		stringstream ss;
		Mat window;
		while(k < images.size() && counter < number){
			Mat img = imread(path + "/" + images[k]);
			if (img.cols != 0 && img.rows != 0) {
				resize(img, img, Size(200, 100));
				for (int j = 0; j < img.rows - size - delta && counter < number; j += delta) {
					for (int i = 0; i < img.cols - size - delta && counter < number; i += delta) {
						window = img(Rect(i, j, size, size));
						ss.str("");
						ss << outputDir << "/image_" << counter << ".pgm";
						imwrite(ss.str(), window);
						counter++;
						cout << "\rGenerated: " << counter << "/" << number << " images" << flush;
					}
				}
			}
			k++;
		}

	}
};



#endif /* FACEDETECTOR_UTILS_UTILS_HPP_ */
