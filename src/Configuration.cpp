/*
								Configuration.cpp

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*
*	This file is part of:	freeture
*
*	Copyright:		(C) 2014-2015 Yoan Audureau
*                               FRIPON-GEOPS-UPSUD-CNRS
*
*	License:		GNU General Public License
*
*	FreeTure is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*	FreeTure is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*	You should have received a copy of the GNU General Public License
*	along with FreeTure. If not, see <http://www.gnu.org/licenses/>.
*
*	Last modified:		20/10/2014
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    Configuration.cpp
* \author  Yoan Audureau -- FRIPON-GEOPS-UPSUD
* \version 1.0
* \date    13/06/2014
* \brief   Get FreeTure's parameters from a configuration file.
*/

#include "Configuration.h"

Configuration::Configuration(void){}

void Configuration::Clear(){

    data.clear();

}

bool Configuration::Load(const string& file){

    ifstream inFile(file.c_str());

    if (!inFile.good()){

        cout << "Cannot read configuration file : " << file << endl;
        return false;
    }

    while (inFile.good() && ! inFile.eof()){

        string line;
        getline(inFile, line);

        // filter out comments
        if (!line.empty()){

            int pos = line.find('#');

            if (pos != string::npos){

                line = line.substr(0, pos);

            }
        }

        // split line into key and value
        if (!line.empty()){

            int pos = line.find('=');

            if (pos != string::npos){

                string key     = Trim(line.substr(0, pos));
                string value   = Trim(line.substr(pos + 1));

                if (!key.empty() && !value.empty()){

                    data[key] = value;

                }
            }
        }
    }

    return true;
}

bool Configuration::Contains(const string& key) const{

    return data.find(key) != data.end();
}

bool Configuration::Get(const string& key, string& value) const{

    map<string,string>::const_iterator iter = data.find(key);

    if(iter != data.end()){

        value = iter->second;
        return true;

    }else{

        return false;
    }
}

bool Configuration::Get(const string& key, int& value) const{

    string str;

    if(Get(key, str)){

        value = atoi(str.c_str());
        return true;

    }else{

        return false;
    }
}

bool Configuration::Get(const string& key, long& value) const{

    string str;

    if(Get(key, str)){

        value = atol(str.c_str());
        return true;

    }else{

        return false;
    }
}

bool Configuration::Get(const string& key, double& value) const{

    string str;

    if(Get(key, str)){

        value = atof(str.c_str());
        return true;

    }else{

        return false;
    }
}

bool Configuration::Get(const string& key, bool& value) const{

    string str;

    if(Get(key, str)){

        value = (str == "true");
        return true;

    }else{

        return false;
    }
}

string Configuration::Trim(const string& str){

    int first = str.find_first_not_of(" \t");

    if(first != string::npos){

        int last = str.find_last_not_of(" \t");

        return str.substr(first, last - first + 1);

    }else{

        return "";
    }
}
