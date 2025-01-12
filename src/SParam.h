#pragma once

/*
                                SParam.h

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*
*   This file is part of:   freeture
*
*   Copyright:      (C) 2014-2016 Yoan Audureau
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
*   Last modified:      20/03/2018
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    SParam.h
* \author  Yoan Audureau -- Chiara Marmo -- GEOPS-UPSUD
* \version 1.2
* \date    20/03/2018
* \brief   FreeTure parameters
*/
//header refactoring ok
#include "Commons.h"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ELogSeverityLevel.h"
#include "ECamPixFmt.h"
#include "ESmtpSecurity.h"
#include "ETimeMode.h"
#include "EImgFormat.h"
#include "EDetMeth.h"
#include "EStackMeth.h"

// #include <fstream>
// #include <string>
// #include <iostream>
// #include <map>
// #include <stdlib.h>
// #include "ECamPixFmt.h"
// #include "ETimeMode.h"
// #include "EImgFormat.h"
// #include "EDetMeth.h"
// #include "ELogSeverityLevel.h"
// #include "EStackMeth.h"
// #include "ESmtpSecurity.h"
// #include <vector>
// #include "EInputDeviceType.h"



// ******************************************************
// ****************** MAIL PARAMETERS *******************
// ******************************************************
#define DEFAULT_CANARIN -1

namespace freeture
{
    struct mailParam {
        bool            MAIL_DETECTION_ENABLED;
        std::string          MAIL_SMTP_SERVER;
        SmtpSecurity    MAIL_CONNECTION_TYPE;
        std::string          MAIL_SMTP_LOGIN;
        std::string          MAIL_SMTP_PASSWORD;
        std::vector<std::string>  MAIL_RECIPIENTS;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // ******************* LOG PARAMETERS *******************
    // ******************************************************

    struct logParam {
        std::string         LOG_PATH;
        int                 LOG_ARCHIVE_DAY;
        int                 LOG_SIZE_LIMIT;
        LogSeverityLevel    LOG_SEVERITY;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // **************** OUTPUT DATA PARAMETERS **************
    // ******************************************************

    struct dataParam {
        std::string  DATA_PATH;
        bool    FITS_COMPRESSION;
        std::string  FITS_COMPRESSION_METHOD;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // **************** INPUT FRAMES PARAMETERS *************
    // ******************************************************

    struct framesParam {
        int INPUT_TIME_INTERVAL;
        std::vector<std::string> INPUT_FRAMES_DIRECTORY_PATH;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // **************** INPUT VIDEO PARAMETERS **************
    // ******************************************************

    struct videoParam {
        int INPUT_TIME_INTERVAL;
        std::vector<std::string> INPUT_VIDEO_PATH;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // ********* SCHEDULED ACQUISITION PARAMETERS ***********
    // ******************************************************

    struct scheduleParam {
        int hours = DEFAULT_CANARIN;
        int min = DEFAULT_CANARIN;
        int sec = DEFAULT_CANARIN;
        int exp = DEFAULT_CANARIN;
        int gain = DEFAULT_CANARIN;
        int rep = DEFAULT_CANARIN;
        CamPixFmt fmt;
    };

    // ******************************************************
    // ************** INPUT CAMERA PARAMETERS ***************
    // ******************************************************
  
    struct cameraParam {
        double      ACQ_FPS = DEFAULT_CANARIN;
        CamPixFmt   ACQ_FORMAT;
        bool        ACQ_RES_CUSTOM_SIZE;
        bool        SHIFT_BITS;
        int         ACQ_NIGHT_EXPOSURE = DEFAULT_CANARIN;
        int         ACQ_NIGHT_GAIN = DEFAULT_CANARIN;
        int         ACQ_DAY_EXPOSURE = DEFAULT_CANARIN;
        int         ACQ_DAY_GAIN = DEFAULT_CANARIN;
        int         ACQ_STARTX = DEFAULT_CANARIN;
        int         ACQ_STARTY = DEFAULT_CANARIN;
        int         ACQ_HEIGHT = DEFAULT_CANARIN;
        int         ACQ_WIDTH = DEFAULT_CANARIN;
        bool        EXPOSURE_CONTROL_ENABLED;
        int         EXPOSURE_CONTROL_FREQUENCY = DEFAULT_CANARIN;
        bool        EXPOSURE_CONTROL_SAVE_IMAGE;
        bool        EXPOSURE_CONTROL_SAVE_INFOS;

        struct ephemeris {
            bool    EPHEMERIS_ENABLED;
            double  SUN_HORIZON_1;
            double  SUN_HORIZON_2;
            std::vector<int>  SUNRISE_TIME;
            std::vector<int>  SUNSET_TIME;
            int     SUNSET_DURATION = DEFAULT_CANARIN;
            int     SUNRISE_DURATION = DEFAULT_CANARIN;
        };
        ephemeris ephem;

        struct regularCaptures {
            bool        ACQ_REGULAR_ENABLED;
            TimeMode    ACQ_REGULAR_MODE;
            std::string      ACQ_REGULAR_PRFX;
            ImgFormat   ACQ_REGULAR_OUTPUT;
            struct regularParam {
                int interval = DEFAULT_CANARIN;
                int exp  = DEFAULT_CANARIN;
                int gain = DEFAULT_CANARIN;
                int rep  = DEFAULT_CANARIN;
                CamPixFmt fmt;
            };
            regularParam ACQ_REGULAR_CFG;
        };
        regularCaptures regcap;

        struct scheduledCaptures {
            bool        ACQ_SCHEDULE_ENABLED;
            ImgFormat   ACQ_SCHEDULE_OUTPUT;
            std::vector<scheduleParam> ACQ_SCHEDULE;
        };
        scheduledCaptures schcap;   

        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // **************** DETECTION PARAMETERS ****************
    // ******************************************************

    struct detectionParam {
        int         ACQ_BUFFER_SIZE;
        bool        ACQ_MASK_ENABLED;
        bool        ACQ_AUTOEXPOSURE_ENABLED;
        std::string ACQ_MASK_PATH;
        cv::Mat     MASK;
        bool        DET_ENABLED;
        TimeMode    DET_MODE;
        bool        DET_DEBUG;
        std::string DET_DEBUG_PATH;
        int         DET_TIME_AROUND;
        int         DET_TIME_MAX;
        DetMeth     DET_METHOD;
        bool        DET_SAVE_FITS3D;
        bool        DET_SAVE_FITS2D;
        bool        DET_SAVE_SUM;
        bool        DET_SUM_REDUCTION;
        StackMeth   DET_SUM_MTHD;
        bool        DET_SAVE_SUM_WITH_HIST_EQUALIZATION;
        bool        DET_SAVE_AVI;
        bool        DET_UPDATE_MASK;
        int         DET_UPDATE_MASK_FREQUENCY;
        bool        DET_DEBUG_UPDATE_MASK;
        bool        DET_DOWNSAMPLE_ENABLED;

        struct detectionMethod1 {
            bool    DET_SAVE_GEMAP;
            bool    DET_SAVE_DIRMAP;
            bool    DET_SAVE_POS;
            int     DET_LE_MAX;
            int     DET_GE_MAX;
            //bool    DET_SAVE_GE_INFOS;
        };
        detectionMethod1 temporal;

        bool status;
        std::vector<std::string> errormsg;

    };

    // ******************************************************
    // ******************* STACK PARAMETERS *****************
    // ******************************************************

    struct stackParam {
        bool        STACK_ENABLED;
        TimeMode    STACK_MODE;
        int         STACK_TIME;
        int         STACK_INTERVAL;
        StackMeth   STACK_MTHD;
        bool        STACK_REDUCTION;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // ****************** STATION PARAMETERS ****************
    // ******************************************************

    struct stationParam {
        std::string STATION_NAME;
        std::string TELESCOP;
        std::string OBSERVER;
        std::string INSTRUMENT;
        std::string CAMERA;
        double FOCAL;
        double APERTURE;
        double SITELONG;
        double SITELAT;
        double SITEELEV;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // ***************** FITS KEYS PARAMETERS ***************
    // ******************************************************

    struct fitskeysParam {
        std::string FILTER;
        double K1;
        double K2;
        std::string COMMENT;
        double CD1_1;
        double CD1_2;
        double CD2_1;
        double CD2_2;
        double XPIXEL;
        double YPIXEL;
        bool status;
        std::vector<std::string> errormsg;
    };

    // ******************************************************
    // ****************** FREETURE PARAMETERS ***************
    // ******************************************************

    struct parameters {
        //std::pair<std::pair<int, bool>,std::string> DEVICE_ID; // Pair : <value, status>
        int             DEVICE_ID;
        std::string     CAMERA_SERIAL;
        bool            CAMERA_INIT;
        std::string     CAMERA_INIT_CONFIG;
        dataParam       data;
        logParam        log;
        framesParam     framesInput;
        videoParam      vidInput;
        cameraParam     camInput;
        detectionParam  det;
        stackParam      st;
        stationParam    station;
        fitskeysParam   fitskeys;
        mailParam       mail;
    };

}