#pragma once
/*
                                    Frame.h

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
* \file    Frame.h
* \author  Yoan Audureau -- Chiara Marmo -- GEOPS-UPSUD
* \version 1.2
* \date    19/03/2018
* \brief   Image container.
*/
#include "Commons.h"
#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

#include "TimeDate.h"
#include "ECamPixFmt.h"

namespace freeture
{
    class Frame {

    private:
        uint8_t* mDataBuffer;
        size_t mSize;
    public:

        TimeDate::Date      mDate;               // Acquisition date.
        double              mExposure;           // Camera's exposure value used to grab the frame.
        int                 mGain;               // Camera's gain value used to grab the frame.
        CamPixFmt           mFormat;             // Pixel format.
        std::string         mFileName;           // Frame's name.
        int                 mFrameNumber;        // Each frame is identified by a number corresponding to the acquisition order.
        int                 mFrameRemaining;     // Define the number of remaining frames if the input source is a video or a set of single frames.
        double              mSaturatedValue;     // Max pixel value in the image.
        int                 mFps;                // Camera's fps.
        int                 mStartX;
        int                 mStartY;
        int                 mWidth;
        int                 mHeight;

        std::shared_ptr<cv::Mat> Image;                // Frame's image data.

        Frame(cv::Mat capImg, int g, double e, std::string acquisitionDate);

        Frame();

        ~Frame();

        uint8_t* getData();

        void SetImage(const uint8_t*, size_t);
    };
}
