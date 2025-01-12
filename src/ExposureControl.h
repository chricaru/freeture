#pragma once
/*
                            ExposureControl.h

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*
*   This file is part of:   freeture
*
*   Copyright:      (C) 2014-2015 Yoan Audureau
*                               GEOPS-UPSUD-CNRS
*
*   License:        GNU General Public License
*
*   FreeTure is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*   FreeTure is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*   You should have received a copy of the GNU General Public License
*   along with FreeTure. If not, see <http://www.gnu.org/licenses/>.
*
*   Last modified:      20/07/2015
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    ExposureControl.h
* \author  Yoan Audureau -- GEOPS-UPSUD
* \version 1.0
* \date    03/06/2014
* \brief   Create/Analyse histogram of a gray image.
*/
#include "Commons.h"
#include <memory>
#include <string>
#include <vector>

// #include <iostream>
// #include "Conversion.h"
// #include "ECamPixFmt.h"
// #include "Device.h"
// #include "SaveImg.h"
// #include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include "TimeDate.h"

namespace freeture
{
    class Device;

    class ExposureControl {
        
    private:

        float           bin_0; // 8 bits : [0,50]       12 bits : [0,819]
        float           bin_1; // 8 bits : ]50,100]     12 bits : ]819,1638]
        float           bin_2; // 8 bits : ]100,150]    12 bits : ]1638,2458]
        float           bin_3; // 8 bits : ]150,200]    12 bits : ]2458,3278]
        float           bin_4; // 8 bits : ]200,255]    12 bits : ]3278,4095]
        cv::Mat         bins;
        double          minCameraExposureValue;
        double          maxCameraExposureValue;
        double          exposureValue;
        bool            autoExposureFinished;
        int             autoExposureTimeInterval;
        int             frameToSkip;
        int             frameSkippedCounter;
        bool            autoExposureInitialized;
        bool            autoExposureSaveImage;
        bool            autoExposureSaveInfos;
        std::string          autoExposureDataLocation;
        std::vector<float>   msvArray_1;
        std::vector<double>  expArray_1;
        std::vector<float>   msvArray_2;
        std::vector<double>  expArray_2;
        bool            incrementExposureTimeValue;
        float           msvMin_1;
        float           msvMax_1;
        double          expMin_1;
        double          expMax_1;
        float           msvMin_2;
        float           msvMax_2;
        double          expMin_2;
        double          expMax_2;
        bool            step1;
        bool            step2;
        std::string          finalDataLocation;
        std::string          stationName;
        double          finalExposureTime;
        int mNbFramesControlled;
        std::string mRefDate;
        long mSecTime;

    public:

        /**
        * Constructor.
        *
        * @param file Path of the configuration file.
        * @param file Path of the configuration file.
        * @param file Path of the configuration file.
        * @param file Path of the configuration file.
        * @param file Path of the configuration file.
        */
        ExposureControl(int timeInterval, bool saveImage, bool saveInfos, std::string dataPath, std::string station);

        bool calculate(std::shared_ptr<cv::Mat> image, cv::Mat& mask);

        float computeMSV();

        bool controlExposureTime(freeture::Device* camera, std::shared_ptr<cv::Mat> image, TimeDate::Date imageDate, cv::Mat mask, double fps);

        bool checkDataLocation(TimeDate::Date date);

    private:

        void clear() { bin_0 = 0; bin_1 = 0; bin_2 = 0; bin_3 = 0; bin_4 = 0; };

    };
}
