#pragma once

/**
* \file    CameraLucidArena.h
* \author  Andrea Novati -- N3 S.r.l.
* \version 1.2
* \date    03/15/2023
* \brief   Use Aravis library to pilot Lucid phoenix cameras.
*
*/

#ifdef LINUX
#include "Commons.h"
#include "CameraDeviceManager.h"

    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/imgproc/imgproc.hpp>

    #include <iostream>
    #include <string>
    #include "Frame.h"
    #include "TimeDate.h"
    #include "Camera.h"
    #include "arv.h"
    #include "arvinterface.h"
    #include <time.h>
    #include <algorithm>
    #include "EParser.h"

    #define BOOST_LOG_DYN_LINK 1

    #include <boost/log/common.hpp>
    #include <boost/log/expressions.hpp>
    #include <boost/log/utility/setup/file.hpp>
    #include <boost/log/utility/setup/console.hpp>
    #include <boost/log/utility/setup/common_attributes.hpp>
    #include <boost/log/attributes/named_scope.hpp>
    #include <boost/log/attributes.hpp>
    #include <boost/log/sinks.hpp>
    #include <boost/log/sources/logger.hpp>
    #include <boost/log/core.hpp>
    #include "ELogSeverityLevel.h"

namespace freeture
{
    class CameraDescription;
    class cameraParam;

    class CameraLucidAravis : public Camera
    {

    private:

        GError* error = nullptr;        // ARAVIS API Error
        ArvCamera* camera;                // Camera to control.
        ArvPixelFormat  pixFormat;              // Image format.
        ArvStream* stream;                // Object for video stream reception.
        int             mStartX;                // Crop starting X.
        int             mStartY;                // Crop starting Y.
        int             mWidth;                 // Camera region's width.
        int             mHeight;                // Camera region's height.
        double          fps;                    // Camera acquisition frequency.
        double          gainMin;                // Camera minimum gain.
        double          gainMax;                // Camera maximum gain.
        unsigned int    payload;                // Width x height.
        double          exposureMin;            // Camera's minimum exposure time.
        double          exposureMax;            // Camera's maximum exposure time.
        double          fpsMin;                 // Camera's minimum frame rate.
        double          fpsMax;                 // Camera's maximum frame rate.
        double          temperature;            // Camera temperature in °C
        const char* capsString;
        int             gain;                   // Camera's gain.
        double          exp;                    // Camera's exposure time.
        bool            shiftBitsImage;         // For example : bits are shifted for dmk's frames.
        guint64         nbCompletedBuffers;     // Number of frames successfully received.
        guint64         nbFailures;             // Number of frames failed to be received.
        guint64         nbUnderruns;
        int             frameCounter;           // Counter of success received frames.


    public:


        CameraLucidArena(CameraDescription, cameraParam );

        CameraLucidArena();

        ~CameraLucidArena();

        bool createDevice(int id);
        bool recreateDevice(int);

        void grabCleanse();

        bool acqStart();

        void acqStop();

        bool grabImage(Frame& newFrame);

        bool grabSingleImage(Frame& frame);

        bool getDeviceNameById(int id, std::string& device);

        void getExposureBounds(double& eMin, double& eMax);

        void getFPSBounds(double& fMin, double& fMax);

        void getGainBounds(double& gMin, double& gMax);

        bool getPixelFormat(CamPixFmt& format);

        bool getFrameSize(int& x, int& y, int& w, int& h);

        bool getFPS(double& value);

        std::string getModelName();

        double getExposureTime();

        bool setAutoExposure(bool val);

        bool setExposureTime(double exp);

        bool setGain(double gain);

        bool setFPS(double fps);

        bool setFrameSize(int startx, int starty, int width, int height, bool customSize);

        bool setPixelFormat(CamPixFmt depth);

        void saveGenicamXml(std::string p);

        bool setSize(int startx, int starty, int width, int height, bool customSize);

        void getAvailablePixelFormats();


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

        double getMinExposureTime() override;

    };
}
#endif
