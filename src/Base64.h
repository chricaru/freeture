#pragma once

/*
                                    Base64.h

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
*   Last modified:      26/11/2014
*
*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**
* \file    Base64.h
* \author  Yoan Audureau -- GEOPS-UPSUD
* \version 1.0
* \date    26/11/2014
* \brief   Handle Base64 encryption.
*/
//header refactoring ok
#include "Commons.h"

#include <string>

namespace freeture
{
    class Base64 {

    public:

        /**
         * Constructor.
         */
        Base64() {};

        /**
         * Destructor.
         */
        ~Base64() {};

        /**
         * Encode string data with base64.
         *
         * @param data String to encode.
         * @return Encoded string.
         */
        static std::string encodeBase64(std::string data);
    };
}