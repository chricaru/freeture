#pragma once
/*
                            DetectionTemplate.h

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
*   Last modified:      03/03/2015
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    DetectionTemplate.h
* \author  Yoan Audureau -- GEOPS-UPSUD
* \version 1.0
* \date    03/03/2015
*/
//header refactoring ok
#include "Commons.h"

#include <string>
#include <vector>

#include "Detection.h"
#include "SParam.h"
#include "TimeDate.h"
#include "ECamPixFmt.h"

// 
// #ifdef LINUX
//     #define BOOST_LOG_DYN_LINK 1
// #endif
// 
// #include <boost/circular_buffer.hpp>
// #include <opencv2/video/tracking.hpp>
// #include <boost/tokenizer.hpp>
// #include <boost/log/common.hpp>
// #include <boost/log/expressions.hpp>
// #include <boost/log/utility/setup/file.hpp>
// #include <boost/log/utility/setup/console.hpp>
// #include <boost/log/utility/setup/common_attributes.hpp>
// #include <boost/log/attributes/named_scope.hpp>
// #include <boost/log/attributes.hpp>
// #include <boost/log/sinks.hpp>
// #include <boost/log/sources/logger.hpp>
// #include <boost/log/core.hpp>
// #include "ELogSeverityLevel.h"
// #include "TimeDate.h"
// #include "Fits2D.h"
// #include "Fits.h"
// #include "Frame.h"
// #include "EStackMeth.h"
// #include "ECamPixFmt.h"
// #include "GlobalEvent.h"
// #include "LocalEvent.h"
// #include "Detection.h"
// #include "EParser.h"
// #include "SaveImg.h"
// #include <vector>
// #include <utility>
// #include <iterator>
// #include <algorithm>
// #include <boost/filesystem.hpp>
// #include "ImgProcessing.h"
// #include "Mask.h"


namespace freeture
{
    class Mask;
    class Frame;

    class DetectionTemplate : public Detection {

    private:


        int                 mImgNum;                // Current frame number.
        cv::Mat                 mPrevFrame;             // Previous frame.
        cv::Mat                 mMask;                  // Mask applied to frames.
        int                 mDataSetCounter;
        detectionParam      mdtp;
        Mask* mMaskControl;


    public:

        DetectionTemplate(detectionParam dtp, CamPixFmt fmt);

        ~DetectionTemplate();

        void initMethod(std::string cfgPath);

        bool runDetection(Frame& c);

        void saveDetectionInfos(std::string p, int nbFramesAround);

        void resetDetection(bool loadNewDataSet);

        void resetMask();

        int getEventFirstFrameNb();

        TimeDate::Date getEventDate();

        int getEventLastFrameNb();

    private:

        void createDebugDirectories(bool cleanDebugDirectory);

    };
}
