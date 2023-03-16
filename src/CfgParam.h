/*
                                CfgParam.h

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
*   Last modified:      20/10/2014
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    CfgParam.h
* \author  Yoan Audureau -- GEOPS-UPSUD
* \version 1.0
* \date    03/06/2014
* \brief   FreeTure parameters
*/

#pragma once

#include "EInputDeviceType.h"
#include "CfgParam.h"
#include <vector>

namespace freeture {

    class CfgParam{

        private :
            static class Init {

                public :

                    Init() {

                        m_Logger.add_attribute("ClassName", boost::log::attributes::constant<std::string>("CfgParam"));

                    }

            }initializer;

            static boost::log::sources::severity_logger< LogSeverityLevel > m_Logger;
            std::vector<string> m_EMsg;

            CfgLoader m_Cfg;
            parameters m_Param;
            InputDeviceType m_InputType;

            void loadDeviceID();
            void loadDataParam();
            void loadLogParam();
            void loadFramesParam();
            void loadVidParam();
            void loadCamParam();
            void loadDetParam();
            void loadStackParam();
            void loadStationParam();
            void loadFitskeysParam();
            void loadMailParam();



        public :

            bool showErrors;

            /**
             * Constructor.
             *
             */
            CfgParam(string cfgFilePath);

            int             getDeviceID();
            dataParam       getDataParam();
            logParam        getLogParam();
            framesParam     getFramesParam();
            videoParam      getVidParam();
            cameraParam     getCamParam();
            detectionParam  getDetParam();
            stackParam      getStackParam();
            stationParam    getStationParam();
            fitskeysParam   getFitskeysParam();
            mailParam       getMailParam();
            parameters      getAllParam();

            bool deviceIdIsCorrect();
            bool dataParamIsCorrect();
            bool logParamIsCorrect();
            bool framesParamIsCorrect();
            bool vidParamIsCorrect();
            bool camParamIsCorrect();
            bool detParamIsCorrect();
            bool stackParamIsCorrect();
            bool stationParamIsCorrect();
            bool fitskeysParamIsCorrect();
            bool mailParamIsCorrect();
            bool allParamAreCorrect();
            bool inputIsCorrect();

    };
}
