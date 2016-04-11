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
#include <vector>

class Utils {
public:
	static std::vector<std::string> open(std::string path = ".") {
	    DIR*    dir;
	    dirent* pdir;
	    std::vector<std::string> files;

	    dir = opendir(path.c_str());

	    while (pdir = readdir(dir)) {
	    	if(strcmp( pdir->d_name, ".") != 0 && strcmp(pdir->d_name, "..") != 0 )
	    		files.push_back(pdir->d_name);
	    }

	    return files;
	}
};



#endif /* FACEDETECTOR_UTILS_UTILS_HPP_ */
