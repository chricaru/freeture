#pragma once
/*
                                OpenSSL.h

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
* \file    OpenSSL.h
* \author  Yoan Audureau -- GEOPS-UPSUD
* \version 1.0
* \date    30/05/2015
*/
#include "Commons.h"

#ifdef LINUX
    #define BOOST_LOG_DYN_LINK 1
#endif

#include <boost/asio.hpp>

#include <openssl/err.h>
#include <openssl/ssl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <ostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "ELogSeverityLevel.h"
#include "Logger.h"

namespace freeture
{

    class OpenSSL {

    public:

        std::unique_ptr< SSL_CTX, decltype(SSL_CTX_free)*> ctx_;
        std::unique_ptr< SSL, decltype(SSL_free)* > ssl_;
        enum {
            errorBufSize = 256,
            readBufSize = 256
        };

        /**
        * Constructor : Create SSL connection.
        *
        * @param socket Network connection.
        */
        OpenSSL(int socket);

        /**
        * Write data on the SSL connection.
        *
        * @param msg Data to write.
        */
        void Write(const std::string& msg);

        /**
        * Read data on the SSL connection.
        *
        * @param isDoneReceiving Struct to handle response.
        * @return Response.
        */
        template<typename IsDoneReceivingFunctorType> std::string Read(IsDoneReceivingFunctorType isDoneReceiving) {

            char buf[readBufSize];
            std::string read;

            while (true) {

                const int rstRead = SSL_read(ssl_.get(), buf, readBufSize);
                if (0 == rstRead) {
                    LOG_ERROR << "Connection lost while read.";
                    throw "Connection lost while read.";
                    //throw runtime_error("Connection lost while read.");
                }
                if (0 > rstRead && SSL_ERROR_WANT_READ == SSL_get_error(ssl_.get(), rstRead))
                    continue;
                read += std::string(buf, buf + rstRead);
                if (isDoneReceiving(read)) return read;
            }
        }

        /**
        * Destructor : Shutdown SSL connection.
        *
        */
        ~OpenSSL();

        /**
        * OpenSSL's library initialization.
        *
        */
        struct StaticInitialize {

            StaticInitialize() {

                ERR_load_crypto_strings();
                SSL_load_error_strings();
                SSL_library_init();
            }

            ~StaticInitialize() {
                ERR_free_strings();
            }
        };
    };
}
