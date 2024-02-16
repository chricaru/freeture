/**
* \file    CameraLucidArena_PHX016S.cpp
* \author  Andrea Novati -- N3 S.r.l.
* \version 1.0
* \date    03/15/2023
* \brief   Use Arena SDK library to pilot Lucid PHX016S Cameras
*
*/
#include "CameraLucidArena_PHX016S.h"

#include <time.h>
#include <algorithm>
#include <iostream>
#include <boost/date_time.hpp>

#include <opencv2/opencv.hpp>

#include <ArenaApi.h>
#include <SaveApi.h>

#include "Logger.h"
#include "ELogSeverityLevel.h"
#include "EParser.h"
#include "Frame.h"
#include "TimeDate.h"
#include "Camera.h"
#include "ArenaSDKManager.h"

using namespace freeture;
using namespace std;

    /**
     * CTor.
     */
    CameraLucidArena_PHX016S::CameraLucidArena_PHX016S(CameraDescription camera_descriptor, cameraParam settings):
        Camera(camera_descriptor, settings),
        m_StartX(settings.ACQ_STARTX), m_StartY(settings.ACQ_STARTY), m_Width(settings.ACQ_WIDTH), m_Height(settings.ACQ_HEIGHT), 
        m_FPS(settings.ACQ_FPS), m_MinFPS(MIN_FPS), m_MaxFPS(MAX_FPS),
        m_Gain(settings.ACQ_DAY_GAIN), m_MinGain(MIN_GAIN), m_MaxGain(MAX_GAIN), 
        m_ExposureTime(settings.ACQ_DAY_EXPOSURE), m_MinExposure(MIN_US_NORMAL), m_MaxExposure(MAX_US_NORMAL),
        payload(0), frameCounter(0), shiftBitsImage(settings.SHIFT_BITS)
    {
        m_ExposureAvailable = true;
        m_GainAvailable = true;
    }

    /**
     * Create a Lucid camera using serial number
     */
    bool CameraLucidArena_PHX016S::createDevice()
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::createDevice";

        Arena::DeviceInfo target_device_info;

        if (!getDeviceInfoBySerial(m_CameraDescriptor.Serial, target_device_info)) {
            LOG_ERROR << "Camera not found";
            return false;
        }

        LOG_DEBUG << "CameraLucidArena_PHX016S::createDevice; "<< "DEVICE NAME " << target_device_info.ModelName() << std::endl;

        m_ArenaDevice = m_ArenaSDKSystem->CreateDevice(target_device_info);

        if(m_ArenaDevice == nullptr)
        {
            LOG_ERROR << "Fail to connect the camera.";
            return false;
        }

        return true;
    }

    void CameraLucidArena_PHX016S::getFPSBounds(double &fMin, double &fMax)
    {
        if (m_ArenaDevice == nullptr)
        {
           LOG_ERROR  << "Camera is nullptr";
            return;
        }

       double fpsMin = 0.0;
       double fpsMax = 0.0;

       GenApi::CFloatPtr pAcquisitionFrameRate = m_ArenaDevice->GetNodeMap()->GetNode("AcquisitionFrameRate");

       if (!pAcquisitionFrameRate)
       {
           LOG_ERROR  << "AcquisitionFrameRateEnable node not found";
            return;
       }

       fpsMax = pAcquisitionFrameRate->GetMax();
       fpsMin = pAcquisitionFrameRate->GetMin();

       fMin = fpsMin;
       fMax = fpsMax;
    }

    bool CameraLucidArena_PHX016S::setFPS(double fps)
    {
        try {
            if (m_ArenaDevice == nullptr)
            {
               LOG_ERROR << "CameraLucidArena_PHX016S::setFPS" << "Camera is nullptr";
                return false;
            }


            bool pAcquisitionFrameRateEnable = m_ArenaDevice->GetNodeMap()->GetNode("AcquisitionFrameRateEnable");

            if (!pAcquisitionFrameRateEnable)
            {
               LOG_ERROR << "CameraLucidArena_PHX016S::setFPS" << "AcquisitionFrameRateEnable node not found";
                return false;
            }

            Arena::SetNodeValue<double>(m_ArenaDevice->GetNodeMap(), "AcquisitionFrameRate", fps);

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setFPS" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setFPS" << "Standard exception thrown: " << ex.what() ;
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setFPS" << "Unexpected exception thrown";
        }

        return false;
    }


    bool CameraLucidArena_PHX016S::getDeviceInfoBySerial(string serial, Arena::DeviceInfo& device_info)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::getDeviceInfoBySerial";
            m_ArenaSDKSystem->UpdateDevices(100);

            vector<Arena::DeviceInfo> deviceInfos = m_ArenaSDKSystem->GetDevices();
            int n_devices = deviceInfos.size();

            for (int i = 0; i < n_devices; i++) {

                if (string(deviceInfos[i].SerialNumber()) == serial) {
                    device_info = deviceInfos[i];
                    return true;
                }
            }
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getDeviceInfoBySerial" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getDeviceInfoBySerial" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getDeviceInfoBySerial" << "Unexpected exception thrown";
        }

        LOG_ERROR << "CameraLucidArena_PHX016S::getDeviceInfoBySerial" << "Fail to retrieve camera with this ID.";
        return false;
    }


    string CameraLucidArena_PHX016S::getModelName()
    {
        Arena::DeviceInfo device_info;

        if (!getDeviceInfoBySerial(m_CameraDescriptor.Serial, device_info)) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getModelName" << "Camera not found";
            return false;
        }

        return device_info.ModelName().c_str();
    }

    void CameraLucidArena_PHX016S::getExposureBounds(double& eMin, double& eMax)
    {
        try {
            if (m_ArenaDevice == nullptr)
            {
               LOG_ERROR << "CameraLucidArena_PHX016S::getModelName" << "Camera is nullptr";
                return;
            }

            double exposureMin = 0.0;
            double exposureMax = 0.0;

            GenApi::CFloatPtr pExposureTime = m_ArenaDevice->GetNodeMap()->GetNode("ExposureTime");

            if (!pExposureTime)
            {
               LOG_ERROR << "CameraLucidArena_PHX016S::getModelName" << "ExposureTime node not found";
                return;
            }



            exposureMax = pExposureTime->GetMax();
            exposureMin = pExposureTime->GetMin();

            eMin = exposureMin;
            eMax = exposureMax;

        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getExposureBounds" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getExposureBounds" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getExposureBounds" << "Unexpected exception thrown";
        }
    }

    double CameraLucidArena_PHX016S::getExposureTime()
    {
        try {
            if (m_ArenaDevice == nullptr)
            {
               LOG_ERROR  <<"CameraLucidArena_PHX016S::getExposureTime" << "Camera is nullptr";
                return -1.0;
            }

            GenApi::CFloatPtr pExposureTime = m_ArenaDevice->GetNodeMap()->GetNode("ExposureTime");

            if (!pExposureTime)
            {
               LOG_ERROR << "CameraLucidArena_PHX016S::getExposureTime" << "ExposureTime node not found";
                return -1;
            }

            double result = pExposureTime->GetValue();

            return result;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getExposureBounds" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getExposureBounds" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getExposureBounds" << "Unexpected exception thrown";
        }

        return -1.0;
    }

    bool CameraLucidArena_PHX016S::setExposureTime(double val)
    {
        try {
            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::setExposureTime; m_ArenaDevice is null";
                return false;
            }

            double expMin, expMax;

            GenApi::CFloatPtr pExposureTime = m_ArenaDevice->GetNodeMap()->GetNode("ExposureTime");

            if (!pExposureTime)
            {
               LOG_ERROR << "CameraLucidArena_PHX016S::setExposureTime; "<< "ExposureTime node not found";
                return false;
            }

            expMax = pExposureTime->GetMax();
            expMin = pExposureTime->GetMin();

            if (val >= expMin && val <= expMax)
            {
                m_ExposureTime = val;

                Arena::SetNodeValue<double>(m_ArenaDevice->GetNodeMap(), "ExposureTime", m_ExposureTime);
            }
            else
            {
               LOG_DEBUG << "CameraLucidArena_PHX016S::setExposureTime; " << "> Exposure value (" << val << ") is not in range [ " << expMin << " - " << expMax << " ]";
                return false;
            }

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setExposureTime" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setExposureTime" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setExposureTime" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::setGain(double val)
    {
        try {
            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::setExposureTime;m_ArenaDevice is null";
                return false;
            }

            double gMin, gMax;

            GenApi::CFloatPtr pGain = m_ArenaDevice->GetNodeMap()->GetNode("Gain");

            if (!pGain)
            {
               LOG_ERROR  << "Gain node not found";
                return false;
            }


            gMax = pGain->GetMax();
            gMin = pGain->GetMin();


            if ((double)val >= gMin && (double)val <= gMax)
            {
                m_Gain = val;
                Arena::SetNodeValue<double>(m_ArenaDevice->GetNodeMap(), "Gain", m_Gain);
            }
            else
            {
                LOG_ERROR << "> Gain value (" << val << ") is not in range [ " << gMin << " - " << gMax << " ]";
                return false;
            }

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setGain" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setGain" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setGain" << "Unexpected exception thrown";
        }


        return false;
    }

    double CameraLucidArena_PHX016S::getGain()
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::getGain";

            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getGain;" << "Gain node not found";
                return false;
            }

            GenApi::CFloatPtr pGain = m_ArenaDevice->GetNodeMap()->GetNode("Gain");

            if (!pGain)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getGain;" << "Gain node not found";
                return false;
            }

            return pGain->GetValue();
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getGain" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getGain" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getGain" << "Unexpected exception thrown";
        }
        return -1;
    }


    void CameraLucidArena_PHX016S::getGainBounds(double& gMin, double& gMax)
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::getGainBounds";
        try {

            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getGainBounds;" << "Camera is nullptr";
                return;
            }

            double gainMin = 0.0;
            double gainMax = 0.0;

            GenApi::CFloatPtr pGain = m_ArenaDevice->GetNodeMap()->GetNode("Gain");

            if (!pGain)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getGainBounds;" << "Gain node not found";
                return;
            }

            gainMax = pGain->GetMax();
            gainMin = pGain->GetMin();

            gMin = gainMin;
            gMax = gainMax;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getGainBounds" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getGainBounds" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getGainBounds" << "Unexpected exception thrown";
        }
    }

    bool CameraLucidArena_PHX016S::getFPS(double& value)
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::getFPS";
        try {
            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getFPS;" << "Camera is nullptr";
                return false;
            }

            GenApi::CFloatPtr pAcquisitionFrameRate = m_ArenaDevice->GetNodeMap()->GetNode("AcquisitionFrameRate");

            if (!pAcquisitionFrameRate)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getFPS;" << "AcquisitionFrameRate node not found";
                return false;
            }

            value = pAcquisitionFrameRate->GetValue();

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFPS" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFPS" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFPS" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::getFrameSize(int& x, int& y, int& w, int& h)
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::getFrameSize";
        try {
            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getFrameSize;" << "Camera is nullptr";
                return false;
            }

            int64_t ww = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Width");
            int64_t hh = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Height");

            int64_t xx = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetX");
            int64_t yy = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetY");

            x = int(xx);
            y = int(yy);
            w = int(ww);
            h = int(hh);

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFrameSize" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFrameSize" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFrameSize" << "Unexpected exception thrown";
        }
        return false;
    }

    bool CameraLucidArena_PHX016S::setCustomFrameSize(int startx, int starty, int width, int height)
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::setCustomFrameSize";
        try {
            int64_t xx = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetX");

            if (xx > 0)
            {
                LOG_INFO << "Starting from : " << m_StartX << "," << m_StartY;
            }
            else
            {
                LOG_WARNING << "OffsetX, OffsetY are not available: cannot set offset.";
                return false;
            }

            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetX", startx);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetY", starty);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Width", width);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Height", height);


            m_StartX = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetX");
            m_StartY = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetY");
            m_Width = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Width");
            m_Height = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Height");


            LOG_INFO << "Camera region size : " << m_Width << "x" << m_Height;

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setCustomFrameSize" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setCustomFrameSize" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getFsetCustomFrameSizePS" << "Unexpected exception thrown";
        }
        return false;
    }

    bool CameraLucidArena_PHX016S::setDefaultFrameSize()
    {

        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setDefaultFrameSize";

            int64_t sensor_width = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "SensorWidth");
            int64_t sensor_height = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "SensorHeigth");

            LOG_INFO << "Camera sensor size : " << sensor_width << "x" << sensor_height;

            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetX", 0);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetY", 0);

            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Width", sensor_width);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Height", sensor_height);

            m_Width = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Width");
            m_Height = Arena::GetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Height");

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDefaultFrameSize" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDefaultFrameSize" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDefaultFrameSize" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::setSize(int startx, int starty, int width, int height, bool customSize)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setSize";


            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::setSize;" << "Camera is nullptr";
                return false;
            }

            if (customSize)
            {
                return setCustomFrameSize(startx, starty, width, height);
            }
            else
            {
                return setDefaultFrameSize();
            }

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setSize" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setSize" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setSize" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::getPixelFormat(CamPixFmt& format)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::getPixelFormat";

            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::getPixelFormat;" << "Camera is nullptr";
                return false;
            }

            GenICam::gcstring _pixFormat = Arena::GetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "PixelFormat");

            string pixFormat(_pixFormat.c_str());

            if (pixFormat == "Mono8")
                format = MONO8;
            else
                if (pixFormat == "Mono12")
                    format = MONO12;
                else
                    if (pixFormat == "Mono16")
                        format = MONO16;
                    else
                        return false;


            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::getPixelFormat" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getPixelFormat" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::getPixelFormat" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::setPixelFormat(CamPixFmt depth)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setPixelFormat";

            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::setPixelFormat;" << "Camera is nullptr";
                return false;
            }


            switch (depth)
            {

            case MONO8:
            {
                Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "PixelFormat", "Mono8");
            }
            break;

            case MONO12:
            {
                Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "PixelFormat", "Mono12");
            }
            break;
            case MONO16:
            {
                Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "PixelFormat", "Mono16");
            }
            break;
            }



            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setPixelFormat" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setPixelFormat" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setPixelFormat" << "Unexpected exception thrown";
        }
        return false;
    }

    bool CameraLucidArena_PHX016S::initOnce()
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::initOnce";

        return true;
    }

    bool CameraLucidArena_PHX016S::initSDK()
    {
        try
        {
            LOG_DEBUG << "CameraLucidArena_PHX016S::initSDK;" << "Retrieve Arena SDK instance";
            m_ArenaSDKSystem = ArenaSDKManager::Get();
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::initSDK" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::initSDK" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::initSDK" << "Unexpected exception thrown";
        }

        return true;
    }

    void CameraLucidArena_PHX016S::getAvailablePixelFormats()
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::getAvailablePixelFormats";

                LOG_INFO << ">> Device pixel formats (firmware ver: 1.65.0 - 01/2023  :";
                LOG_INFO<< "- Mono8";
                LOG_INFO<< "- Mono10";
                LOG_INFO<< "- Mono10p";
                LOG_INFO<< "- Mono10Packed";
                LOG_INFO<< "- Mono12";
                LOG_INFO<< "- Mono12p";
                LOG_INFO<< "- Mono12Packed";
                LOG_INFO<< "- Mono16";
                LOG_INFO<< "- Mono24";
                LOG_INFO<< "- PolarizeMono8";
                LOG_INFO<< "- PolarizeMono12";
                LOG_INFO<< "- PolarizeMono12p";
                LOG_INFO<< "- PolarizeMono12Packed";
                LOG_INFO<< "- PolarizeMono16";
                LOG_INFO<< "- BayerGR8";
                LOG_INFO<< "- BayerRG8";
                LOG_INFO<< "- BayerGB8";
                LOG_INFO<< "- BayerBG8";
                LOG_INFO<< "- BayerGR10";
                LOG_INFO<< "- BayerRG10";
                LOG_INFO<< "- BayerGB10";
                LOG_INFO<< "- BayerBG10";
                LOG_INFO<< "- BayerGR10p";
                LOG_INFO<< "- BayerRG10p";
                LOG_INFO<< "- BayerGB10p";
                LOG_INFO<< "- BayerBG10p";
                LOG_INFO<< "- BayerGR10Packed";
                LOG_INFO<< "- BayerRG10Packed";
                LOG_INFO<< "- BayerGB10Packed";
                LOG_INFO<< "- BayerBG10Packed";
                LOG_INFO<< "- BayerGR12";
                LOG_INFO<< "- BayerRG12";
                LOG_INFO<< "- BayerGB12";
                LOG_INFO<< "- BayerBG12";
                LOG_INFO<< "- BayerGR12p";
                LOG_INFO<< "- BayerRG12p";
                LOG_INFO<< "- BayerGB12p";
                LOG_INFO<< "- BayerBG12p";
                LOG_INFO<< "- BayerGR12Packed";
                LOG_INFO<< "- BayerRG12Packed";
                LOG_INFO<< "- BayerGB12Packed";
                LOG_INFO<< "- BayerBG12Packed";
                LOG_INFO<< "- BayerGR16";
                LOG_INFO<< "- BayerRG16";
                LOG_INFO<< "- BayerGB16";
                LOG_INFO<< "- BayerBG16";
                LOG_INFO<< "- BayerGR24";
                LOG_INFO<< "- BayerRG24";
                LOG_INFO<< "- BayerGB24";
                LOG_INFO<< "- BayerBG24";
                LOG_INFO<< "- RGB8";
                LOG_INFO<< "- RGB12p";
                LOG_INFO<< "- RGB16";
                LOG_INFO<< "- RGB24";
                LOG_INFO<< "- BGR8";
                LOG_INFO<< "- BGR12p";
                LOG_INFO<< "- BGR16";
                LOG_INFO<< "- BGR24";
                LOG_INFO<< "- YCbCr8";
                LOG_INFO<< "- YCbCr8_CbYCr";
                LOG_INFO<< "- YUV422_8";
                LOG_INFO<< "- YUV422_8_UYVY";
                LOG_INFO<< "- YCbCr411_8";
                LOG_INFO<< "- YUV411_8_UYYVYY";
                LOG_INFO<< "- PolarizedAngles_0d_45d_90d_135d_Mono8";
                LOG_INFO<< "- PolarizedAngles_0d_45d_90d_135d_Mono12p";
                LOG_INFO<< "- PolarizedAngles_0d_45d_90d_135d_Mono16";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_Mono8";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_Mono12p";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_Mono16";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_S3_Mono8";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_S3_Mono12p";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_S3_Mono16";
                LOG_INFO<< "- PolarizedDolpAolp_Mono8";
                LOG_INFO<< "- PolarizedDolpAolp_Mono12p";
                LOG_INFO<< "- PolarizedDolp_Mono8";
                LOG_INFO<< "- PolarizedDolp_Mono12p";
                LOG_INFO<< "- PolarizedAolp_Mono8";
                LOG_INFO<< "- PolarizedAolp_Mono12p";
                LOG_INFO<< "- PolarizedAngles_0d_45d_90d_135d_BayerRG8";
                LOG_INFO<< "- PolarizedAngles_0d_45d_90d_135d_BayerRG12p";
                LOG_INFO<< "- PolarizedAngles_0d_45d_90d_135d_BayerRG16";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_BayerRG8";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_BayerRG12p";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_BayerRG16";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_S3_BayerRG8";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_S3_BayerRG12p";
                LOG_INFO<< "- PolarizedStokes_S0_S1_S2_S3_BayerRG16";
                LOG_INFO<< "- PolarizedDolpAolp_BayerRG8";
                LOG_INFO<< "- PolarizedDolpAolp_BayerRG12p";
                LOG_INFO<< "- PolarizedDolp_BayerRG8";
                LOG_INFO<< "- PolarizedDolp_BayerRG12p";
                LOG_INFO<< "- PolarizedAolp_BayerRG8";
                LOG_INFO<< "- PolarizedAolp_BayerRG12p";
                LOG_INFO<< "- Raw24";
                LOG_INFO<< "- Raw16";
                LOG_INFO<< "- YCbCr422_8_CbYCrY";
                LOG_INFO<< "- YCbCr422_16_CbYCrY";
                LOG_INFO<< "- YCbCr422_24_CbYCrY";
                LOG_INFO<< "- YCbCr411_8_CbYYCrYY";
                LOG_INFO<< "- YCbCr411_16_CbYYCrYY";
                LOG_INFO<< "- YCbCr411_24_CbYYCrYY";
                LOG_INFO<< "- PolarizedDolpAngle_Mono8";
                LOG_INFO<< "- PolarizedDolpAngle_Mono12p";
                LOG_INFO<< "- PolarizedDolpAngle_Mono16";
                LOG_INFO<< "- PolarizedDolpAngle_BayerRG8";
                LOG_INFO<< "- PolarizedDolpAngle_BayerRG12p";
                LOG_INFO<< "- PolarizedDolpAngle_BayerRG16";

                // Compare found pixel formats to currently formats supported by freeture

                LOG_INFO<< endl <<  ">> Available pixel formats :";
                LOG_INFO<< "- MONO8 available --> ID : Mono8 "<< endl;
                LOG_INFO<< "- MONO12 available --> ID : Mono12 ";
                LOG_INFO<< "- MONO16 available --> ID : Mono16";

    }

    void CameraLucidArena_PHX016S::grabCleanse()
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::grabCleanse; *** DO NOTHING ***";
    }

    void CameraLucidArena_PHX016S::acqStop()
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::acqStop";

            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::acqStop;" << "Camera is nullptr";
                return;
            }

            //arv_stream_get_statistics(stream, &nbCompletedBuffers, &nbFailures, &nbUnderruns);

            //cout << "Completed buffers = " << (unsigned long long) nbCompletedBuffers  ;
            //cout << "Failures          = " << (unsigned long long) nbFailures          ;
            //cout << "Underruns         = " << (unsigned long long) nbUnderruns         ;

            //LOG_INFO << "Completed buffers = " << (unsigned long long) nbCompletedBuffers;
            //LOG_INFO << "Failures          = " << (unsigned long long) nbFailures;
            //LOG_INFO << "Underruns         = " << (unsigned long long) nbUnderruns;

            LOG_INFO << "Stopping acquisition...";

            m_ArenaDevice->StopStream();
            m_Streaming = false;
            LOG_INFO << "Acquisition stopped.";
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::acqStop" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::acqStop" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::acqStop" << "Unexpected exception thrown";
        }
    }

    /// <summary>
    /// Set continuous mode ans set FPS to current rate
    /// </summary>
    bool CameraLucidArena_PHX016S::setContinuousMode() {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setContinuousMode";
            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "AcquisitionMode", "Continuous");
            setFPS(m_CameraSettings.ACQ_FPS);
            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setContinuousMode" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setContinuousMode" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setContinuousMode" << "Unexpected exception thrown";
        }
        return false;
    }

    /// <summary>
    /// Set single shot mode and MIN_FPS (0.1)
    /// </summary>
    bool CameraLucidArena_PHX016S::setSingleShotMode()
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setSingleShotMode";

            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "AcquisitionMode", "SingleFrame");
            
            if (!setFPS(MIN_FPS))
                return false;

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setSingleShotMode" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setSingleShotMode" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setSingleShotMode" << "Unexpected exception thrown";
        }
        return false;
    }


    bool  CameraLucidArena_PHX016S::setDayContinuous()
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setDayContinuous";

            setContinuousMode();
            setGain(m_CameraSettings.ACQ_DAY_GAIN);
            setExposureTime(m_CameraSettings.ACQ_DAY_EXPOSURE);
            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDayContinuous" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDayContinuous" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDayContinuous" << "Unexpected exception thrown";
        }

        return false;
    }

    bool  CameraLucidArena_PHX016S::setNightContinuous() {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setNightContinuous";

            setContinuousMode();
            setGain(m_CameraSettings.ACQ_NIGHT_GAIN);
            setExposureTime(m_CameraSettings.ACQ_NIGHT_EXPOSURE);
            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setNightContinuous" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setNightContinuous" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setNightContinuous" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::setDayRegular() {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setDayRegular";

            setSingleShotMode();
            setGain(m_CameraSettings.ACQ_DAY_GAIN);
            setExposureTime(m_CameraSettings.ACQ_DAY_EXPOSURE);
            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDayRegular" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDayRegular" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setDayRegular" << "Unexpected exception thrown";
        }

        return false;
    }

    bool  CameraLucidArena_PHX016S::setNightRegular() {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setNightRegular";

            setSingleShotMode();
            setGain(m_CameraSettings.regcap.ACQ_REGULAR_CFG.gain);
            setExposureTime(m_CameraSettings.regcap.ACQ_REGULAR_CFG.exp);
            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setNightRegular" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setNightRegular" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setNightRegular" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::acqStart()
    {
        return acqStart(true);
    }

    bool CameraLucidArena_PHX016S::acqStart(bool continuous)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::acqStart";

            if (m_ArenaDevice == nullptr)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::acqStart;" << "Camera is nullptr";
                return false;
            }

            if (continuous) {
                // Set acquisition mode
                //    Set acquisition mode before starting the stream. Starting the stream
                //    requires the acquisition mode to be set beforehand. The acquisition
                //    mode controls the number of images a device acquires once the stream
                //    has been started. Setting the acquisition mode to 'Continuous' keeps
                //    the stream from stopping. This example returns the camera to its
                //    initial acquisition mode near the end of the example.
                LOG_INFO << "Set camera to CONTINUOUS MODE";
                Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "AcquisitionMode", "Continuous");
            }
            else {
                LOG_INFO << "Set camera to SINGLEFRAME";
                Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "AcquisitionMode", "SingleFrame");
            }
            // Set buffer handling mode
            //    Set buffer handling mode before starting the stream. Starting the
            //    stream requires the buffer handling mode to be set beforehand. The
            //    buffer handling mode determines the order and behavior of buffers in
            //    the underlying stream engine. Setting the buffer handling mode to
            // 
            //    'NewestOnly' ensures the most recent image is delivered, even if it
            //    means skipping frames.
            LOG_INFO << "Set buffer handling mode to 'NewestOnly'";
            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetTLStreamNodeMap(), "StreamBufferHandlingMode", "NewestOnly");

            // Enable stream auto negotiate packet size
            //    Setting the stream packet size is done before starting the stream.
            //    Setting the stream to automatically negotiate packet size instructs
            //    the camera to receive the largest packet size that the system will
            //    allow. This generally increases frame rate and results in fewer
            //    interrupts per image, thereby reducing CPU load on the host system.
            //    Ethernet settings may also be manually changed to allow for a
            //    larger packet size.
            LOG_INFO << "Enable stream to auto negotiate packet size";

            // Disable stream packet resend
            //    Enable stream packet resend before starting the stream. Images are
            //    sent from the camera to the host in packets using UDP protocol,
            //    which includes a header image number, packet number, and timestamp
            //    information. If a packet is missed while receiving an image, a
            //    packet resend is requested and this information is used to retrieve
            //    and redeliver the missing packet in the correct order.
            LOG_INFO << "Disable stream packet resend";
            Arena::SetNodeValue<bool>(m_ArenaDevice->GetTLStreamNodeMap(), "StreamPacketResendEnable", false);


            Arena::SetNodeValue<bool>(m_ArenaDevice->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);

            LOG_INFO << "Set camera TriggerMode to Off";
            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "TriggerMode", "Off");

            // Start stream
            //    Start the stream before grabbing any images. Starting the stream
            //    allocates buffers, which can be passed in as an argument (default: 10),
            //    and begins filling them with data. Starting the stream blocks write
            //    access to many features such as width, height, and pixel format, as
            //    well as acquisition and buffer handling modes, among others. The stream
            //    needs to be stopped later.
            LOG_INFO << "Start acquisition on camera";
            m_ArenaDevice->StartStream();
            m_Streaming = false;

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::acqStart" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::acqStart" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::acqStart" << "Unexpected exception thrown";
        }

        return false;
    }

    bool CameraLucidArena_PHX016S::init()
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::init";
        getTemperature("Sensor");
        return true;
    }

    bool CameraLucidArena_PHX016S::grabInitialization()
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::grabInitialization";

            frameCounter = 0;

            LOG_INFO << "Camera payload (NOT USED): " << payload;

            LOG_INFO << "DEVICE SELECTED : " << m_CameraDescriptor.Id;
            LOG_INFO << "DEVICE DESCRIPTION: " << m_CameraDescriptor.Description;
            LOG_INFO << "DEVICE VENDOR   : " << "Lucid";

            LOG_INFO << "PAYLOAD         : " << payload;
            LOG_INFO << "Start X         : " << m_StartX;
            LOG_INFO << "Start Y         : " << m_StartY;
            LOG_INFO << "Width           : " << m_Width;
            LOG_INFO << "Height          : " << m_Height;
            LOG_INFO << "Exp Range       : [" << m_MinExposure << " - " << m_MaxExposure << "]";
            LOG_INFO << "Exp             : " << m_ExposureTime;
            LOG_INFO << "Gain Range      : [" << m_MinGain << " - " << m_MaxGain << "]";
            LOG_INFO << "Gain            : " << m_Gain;
            LOG_INFO << "Fps             : " << m_FPS;
            LOG_INFO << "Type            : " << m_PixelFormat;

            setGain(m_Gain);
            setFPS(m_FPS);
            setExposureTime(m_ExposureTime);

            // Set buffer handling mode
            //    Set buffer handling mode before starting the stream. Starting the
            //    stream requires the buffer handling mode to be set beforehand. The
            //    buffer handling mode determines the order and behavior of buffers in
            //    the underlying stream engine. Setting the buffer handling mode to
            //    'NewestOnly' ensures the most recent image is delivered, even if it
            //    means skipping frames.

            LOG_INFO << "Set buffer handling mode to 'NewestOnly'";
            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetTLStreamNodeMap(), "StreamBufferHandlingMode", "NewestOnly");

            // Enable stream auto negotiate packet size
            //    Setting the stream packet size is done before starting the stream.
            //    Setting the stream to automatically negotiate packet size instructs
            //    the camera to receive the largest packet size that the system will
            //    allow. This generally increases frame rate and results in fewer
            //    interrupts per image, thereby reducing CPU load on the host system.
            //    Ethernet settings may also be manually changed to allow for a
            //    larger packet size.

            LOG_INFO << "Enable stream to auto negotiate packet size";
            Arena::SetNodeValue<bool>(m_ArenaDevice->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);

            // Disable stream packet resend
            //    Enable stream packet resend before starting the stream. Images are
            //    sent from the camera to the host in packets using UDP protocol,
            //    which includes a header image number, packet number, and timestamp
            //    information. If a packet is missed while receiving an image, a
            //    packet resend is requested and this information is used to retrieve
            //    and redeliver the missing packet in the correct order.

            LOG_INFO << "Disable stream packet resend";
            Arena::SetNodeValue<bool>(m_ArenaDevice->GetTLStreamNodeMap(), "StreamPacketResendEnable", false);

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::grabInitialization" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::grabInitialization" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::grabInitialization" << "Unexpected exception thrown";
        }
        return false;
    }


    bool CameraLucidArena_PHX016S::grabSingleImage(Frame& frame)
    {
        bool res = true;

        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::grabSingleImage";

            if (!setNightRegular())
                return false;
            
            m_FPS = MIN_FPS;
            m_PixelFormat = frame.mFormat;
            m_ExposureTime = frame.mExposure;
            m_Gain = frame.mGain;

            if (!grabInitialization())
                throw "FAILED TO grabInitialization";
            
            if (!acqStart(false))
                throw "FAILED TO acqStart";

            if (!m_ArenaDevice)
                throw "DEVICE DIED";

            Arena::IImage* pImage = m_ArenaDevice->GetImage(IMAGE_TIMEOUT);
            size_t size = pImage->GetSizeFilled();
            size_t width = pImage->GetWidth();
            size_t height = pImage->GetHeight();
            m_PixelFormat = GetPixelFormatName(static_cast<PfncFormat>(pImage->GetPixelFormat()));
            uint64_t timestampNs = pImage->GetTimestampNs();
           
            LOG_DEBUG << " (" << " Gain " << m_Gain << "; FPS " << m_FPS << "; Exposure " << m_ExposureTime << "; " << size << " bytes; " << width << "x" << height << "; " << m_PixelFormat << "; timestamp (ns): " << timestampNs << ")";

            const uint8_t* u_buffer_data = pImage->GetData();

            char* buffer_data = to_char_ptr(u_buffer_data);

            size_t buffer_size = pImage->GetSizeFilled();

            CopyFrame(frame, buffer_data);
          
            // Requeue image buffer
            //    Requeue an image buffer when access to it is no longer needed.
            //    Notice that failing to requeue buffers may cause memory to leak and
            //    may also result in the stream engine being starved due to there
            //    being no available buffers.
            m_ArenaDevice->RequeueBuffer(pImage);
        }
        catch (GenICam::GenericException& e) {
            res = false;
            LOG_ERROR << "CameraLucidArena_PHX016S::grabSingleImage; " << e.what();
        }
        catch (std::exception& ex)
        {
            res = false;
            LOG_ERROR << "CameraLucidArena_PHX016S::grabSingleImage; " << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            res = false;
            LOG_ERROR << "CameraLucidArena_PHX016S::grabSingleImage; " << "Unexpected exception thrown";
        }

        try
        {
          
            LOG_DEBUG << "CameraLucidArena_PHX016S::grabSingleImage" << "Stop Stream";
            m_ArenaDevice->StopStream();
            return res;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::grabSingleImage; " << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::grabSingleImage; " << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::grabSingleImage; " << "Unexpected exception thrown";
        }

        return false;
    }

    /**
     * DTor.
     */
    CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S()
    {
        try {

            if (m_Streaming)
                acqStop();

            m_ArenaSDKSystem->DestroyDevice(m_ArenaDevice);
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S; " << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S; " << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S;" << "Unexpected exception thrown";
        }
    }

    bool CameraLucidArena_PHX016S::grabImage(Frame& newFrame)
    {
        try
        {
            // LOG_DEBUG << "CameraLucidArena_PHX016S::grabImage"; // !!!! LOG SPAM !!!!

            Arena::IImage* pImage = m_ArenaDevice->GetImage(IMAGE_TIMEOUT);

            size_t size = pImage->GetSizeFilled();
            size_t width = pImage->GetWidth();
            size_t height = pImage->GetHeight();

            m_PixelFormat = GetPixelFormatName(static_cast<PfncFormat>(pImage->GetPixelFormat()));

            uint64_t timestampNs = pImage->GetTimestampNs();

            LOG_DEBUG << " ("<<" Gain " << m_Gain << "; FPS " << m_FPS << "; Exposure " << m_ExposureTime << "; " << size << " bytes; " << width << "x" << height << "; " << m_PixelFormat << "; timestamp (ns): " << timestampNs << ")";

            const uint8_t* u_buffer_data = pImage->GetData();

            char* buffer_data = to_char_ptr(u_buffer_data);

            size_t buffer_size = pImage->GetSizeFilled();
           
            CopyFrame(newFrame, buffer_data);

            newFrame.mFrameNumber = frameCounter++;

            m_ArenaDevice->RequeueBuffer(pImage);

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S;" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S;" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::~CameraLucidArena_PHX016S;" << "Unexpected exception thrown";
        }
        return false;
    }

    bool CameraLucidArena_PHX016S::setFrameSize(unsigned int width, unsigned int height, unsigned int startx, unsigned int starty)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::setFrameSize";

            if (startx % 2 != 0)
            {
                LOG_ERROR << "CameraLucidArena_PHX016S::setFrameSize" << "X Offset need to be odd";
                return false;
            }


            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetX", startx);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "OffsetY", starty);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Width", width);
            Arena::SetNodeValue<int64_t>(m_ArenaDevice->GetNodeMap(), "Height", height);

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::setFrameSize" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setFrameSize" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::setFrameSize" << "Unexpected exception thrown";
        }

        return false;
    }


    bool CameraLucidArena_PHX016S::configurationCheck(parameters&)
    {
        try {
            LOG_DEBUG << "CameraLucidArena_PHX016S::configurationCheck";
            //SET CONTINUOUS MODE 
            if (!setDayContinuous()) {
                LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Set DAY CONTINUOUS configuration failed";
                return false;
            }


            if (!setNightContinuous()) {
                LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Set NIGHT REGULAR configuration failed";
                return false;

            }

            //SET SINGLESHOT MODE
            if (!setDayRegular()) {
                LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Set DAY REGULAR configuration failed";
                return false;
            }

            if (!setNightRegular()) {
                LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Set NIGHT REGULAR configuration failed";
                return false;

            }

            return true;
        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Unexpected exception thrown";
        }

        return false;
    }

    void CameraLucidArena_PHX016S::configure(parameters&)
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::configure";
    }

    void CameraLucidArena_PHX016S::fetchBounds(parameters&)
    {
        LOG_DEBUG << "CameraLucidArena_PHX016S::fetchBounds";
        //set pixel format
        setPixelFormat(m_CameraSettings.ACQ_FORMAT);
        
        //set frame size
        setFrameSize(m_Width, m_Height, m_StartX, m_StartY);

        //fetch gain bounds
        getGainBounds(m_MinGain,m_MaxGain);

        //fetch fps bounds
        getFPSBounds(m_MinFPS,m_MaxFPS);

        //fetch exposure bounds
        getExposureBounds(m_MinExposure,m_MaxExposure);
    }

    /// <summary>
    ///  DeviceTemperatureSelector
    ///  *DeviceTemperature
    ///    EnumEntry : 'Sensor1' (Not available)
    ///    EnumEntry : 'Sensor0' (Not available)
    ///    EnumEntry : 'TEC' (Not available)
    ///    EnumEntry : 'Sensor'
    /// </summary>
    /// <returns></returns>
    bool CameraLucidArena_PHX016S::getTemperature(string selector)
    {
        try
        {
            GenICam::gcstring value = GenICam::gcstring(selector.c_str());

            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "AcquisitionMode", "Continuous");
            Arena::SetNodeValue<GenICam::gcstring>(m_ArenaDevice->GetNodeMap(), "DeviceTemperatureSelector", value);

            //sensor
            GenApi::CEnumerationPtr  pEnumeration = m_ArenaDevice->GetNodeMap()->GetNode("DeviceTemperatureSelector");
            GenApi::CEnumEntryPtr pCurrentEntry = pEnumeration->GetCurrentEntry();
            GenICam::gcstring currentEntrySymbolic = pCurrentEntry->GetSymbolic();

            GenApi::CFloatPtr t = m_ArenaDevice->GetNodeMap()->GetNode("DeviceTemperature");

            double temperature = t->GetValue();
            LOG_INFO << "CameraLucidArena_PHX016S::getTemperature;" << currentEntrySymbolic << "=" << temperature << "�C";



        }
        catch (GenICam::GenericException& e) {
            LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << e.what();
        }
        catch (std::exception& ex)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Standard exception thrown: " << ex.what();
        }
        catch (...)
        {
            LOG_ERROR << "CameraLucidArena_PHX016S::configurationCheck" << "Unexpected exception thrown";
        }

        return false;
    }

    void CameraLucidArena_PHX016S::CopyFrame(Frame& newFrame, char* buffer_data)
    {
        boost::posix_time::ptime time = boost::posix_time::microsec_clock::universal_time();
        string acquisitionDate = to_iso_extended_string(time);

        cv::Mat image;
        CamPixFmt imgDepth = MONO8;
        int saturateVal = 0;

        if (m_PixelFormat == "Mono8")
        {
            //BOOST_LOG_SEV(logger, normal) << "Creating Mat 8 bits ...";
            image = cv::Mat(m_Height, m_Width, CV_8UC1, buffer_data);
            imgDepth = MONO8;
            saturateVal = 255;

        }
        else if (m_PixelFormat == "Mono12")
        {

            //BOOST_LOG_SEV(logger, normal) << "Creating Mat 16 bits ...";
            image = cv::Mat(m_Height, m_Width, CV_16UC1, buffer_data);
            imgDepth = MONO12;
            saturateVal = 4095;

            if (shiftBitsImage) {

                unsigned short* p;

                for (int i = 0; i < image.rows; i++) {
                    p = image.ptr<unsigned short>(i);
                    for (int j = 0; j < image.cols; j++)
                        p[j] = p[j] >> 4;
                }
            }
        }
        else if (m_PixelFormat == "Mono16")
        {
            image = cv::Mat(m_Height, m_Width, CV_16UC1, buffer_data);
            imgDepth = MONO16;
            saturateVal = 65535;
        }

        newFrame = Frame(image, m_Gain, m_ExposureTime, acquisitionDate);

        newFrame.mFps = m_FPS;
        newFrame.mFormat = imgDepth;
        newFrame.mSaturatedValue = saturateVal;
        newFrame.mFrameNumber = frameCounter;
        newFrame.mDate = TimeDate::splitIsoExtendedDate(to_iso_extended_string(time));
    }
