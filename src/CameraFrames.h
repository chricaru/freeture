#pragma once
/*
                            CameraFrames.h

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*
*   This file is part of:   freeture
*
*   Copyright:      (C) 2014-2015 Yoan Audureau
*                       2018 Chiara Marmo
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
*   Last modified:      19/03/2018
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    CameraFrames.h
* \author  Yoan Audureau -- Chiara Marmo -- GEOPS-UPSUD
* \version 1.2
* \date    19/03/2018
* \brief   Fits frames in input of acquisition thread.
*/

// #include "ELogSeverityLevel.h"
// #include "Conversion.h"
// #include "TimeDate.h"
// #include "Frame.h"
// #include "Fits2D.h"
// #include "Fits.h"
// #include <list>
// #include <iterator>
// #include <boost/filesystem.hpp>
// #include <boost/tokenizer.hpp>

//headers refactoring ok
#include "Commons.h"

#include <vector>
#include <string>

#include "Camera.h"

using namespace boost::posix_time;

namespace freeture
{
    class CameraDescription;
    class cameraParam;

    class CameraFrames : public Camera {

    private:

        bool searchMinMaxFramesNumber(std::string location);

        std::vector<std::string> mFramesDir;  // List of frames directories to process.
        int mNumFramePos;           // Position of the frame number in its filename.
        int mFirstFrameNum;         // First frame number in a directory.
        int mLastFrameNum;          // Last frame number in a directory.
        bool mReadDataStatus;       // Signal the end of reading data in a directory.
        int mCurrDirId;             // Id of the directory to use.
        std::string mCurrDir;            // Path of the directory to use.

    public:

        CameraFrames(CameraDescription, cameraParam, std::vector<std::string> , int , bool );

        ~CameraFrames();

        bool acqStart() { return true; };


        bool grabImage(Frame& img);

        bool getStopStatus();

        bool loadNextDataSet(std::string& location);

        bool getDataSetStatus();

        bool getFPS(double& value);

        bool setExposureTime(double exp) { return true; };

        bool setGain(double gain) { return true; };

        bool setFPS(double fps) { return true; };

        bool setPixelFormat(CamPixFmt format) { return true; };

        bool setSize(int startx, int starty, int width, int height, bool customSize) { return true; };

        bool getCameraName();


        //ABSTRACT FACTORY METHODS
        /// <summary>
        /// initialize SDK
        /// </summary>
        /// <returns></returns>
        bool initSDK() override;

        /// <summary>
        /// init once, run configuration once (use configuration file)
        /// </summary>
        /// <returns></returns>
        bool initOnce() override;

        bool createDevice() override;

        /// <summary>
        /// init the camera, eg. running functions when created 
        /// CALL GRAB INITIALIZATION 
        /// </summary>
        /// <returns></returns>
        bool init() override;

        /// <summary>
        /// DEPRECATED USE INIT INSTEAD.
        /// 
        /// </summary>
        /// <returns></returns>
        bool grabInitialization() override;

        /// <summary>
        /// retreive main camera boundaries upon configuration: 
        ///     - fps
        ///     - gain
        ///     - exposure time
        /// </summary>
        void fetchBounds(parameters&) override;

        /// <summary>
        /// configure the camera with the given parameters
        /// </summary>
        /// <param name=""></param>
        void configure(parameters&) override;

        /// <summary>
        /// check if configuration is allowed
        /// </summary>
        /// <param name=""></param>
        /// <returns></returns>
        bool configurationCheck(parameters&) override;


    };

}