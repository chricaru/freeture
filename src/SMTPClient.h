#pragma once
/*
                                SMTPClient.h

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
* \file    SMTPClient.h
* \author  Yoan Audureau -- GEOPS-UPSUD
* \version 1.0
* \date    03/12/2014
* \brief   SMTP connection and send mails.
*/

#include "Commons.h"

#include <string>
#include <vector>
#include "ESmtpSecurity.h"

#include "OpenSSL.h"
#include "Logger.h"

#include <boost/asio.hpp>


namespace freeture
{

    class SMTPClient {

    public:

        /**
        * Send mail.
        *
        * @param server SMTP server name.
        * @param login Login to use if a secured connection to the SMTP server is required.
        * @param password Password to use if a secured connection to the SMTP server is required.
        * @param from Mail sender.
        * @param to Mail recipients.
        * @param subject Mail subject.
        * @param message Mail message.
        * @param pathAttachments Path of files to send.
        * @param imgInline
        * @param securityType Use secured connection or not.
        */
        static void sendMail(std::string            server,
            std::string            login,
            std::string            password,
            std::string            from,
            std::vector<std::string>    to,
            std::string            subject,
            std::string            message,
            std::vector<std::string>    pathAttachments,
            SmtpSecurity      securityType);

    private:

        /**
        * Check SMTP answer.
        *
        * @param responseWaited
        * @param socket
        * @return Answer is correct or not.
        */
        static bool checkSMTPAnswer(const std::string& responseWaited, boost::asio::ip::tcp::socket& socket);

        /**
        * Send data to SMTP.
        *
        * @param data Data to send.
        * @param expectedAnswer
        * @param checkAnswer
        * @param printCmd
        */
        static void write(std::string data, std::string expectedAnswer, bool checkAnswer, boost::asio::ip::tcp::socket& socket);

        /**
        * Create MIME message.
        *
        * @return Final message to send.
        */
        static std::string buildMessage(std::string msg, std::vector<std::string> mMailAttachments,
            std::vector<std::string> mMailTo, std::string mMailFrom, std::string mMailSubject);

        /**
        * Get file content.
        *
        * @param filename
        * @return File's content.
        */
        static bool getFileContents(const char* filename, std::string& content);


        struct ReceiveFunctor {

            enum { codeLength = 3 };
            const std::string code;

            ReceiveFunctor(int expectingCode) : code(std::to_string(expectingCode)) {
                if (code.length() != codeLength) {
                    LOG_ERROR << "SMTP code must be three-digits.";
                    throw "SMTP code must be three-digits.";
                    //throw runtime_error("SMTP code must be three-digits.");}
                }
            }

            bool operator()(const std::string& msg) const {

                if (msg.length() < codeLength) return false;
                if (code != msg.substr(0, codeLength)) {
                    LOG_ERROR << "SMTP code must be three-digits.";
                    throw "SMTP code must be three-digits.";
                    //throw runtime_error("SMTP code is not received");
                }

                const size_t posNewline = msg.find_first_of("\n", codeLength);
                if (posNewline == std::string::npos) return false;
                if (msg.at(codeLength) == ' ') return true;
                if (msg.at(codeLength) == '-') return this->operator()(msg.substr(posNewline + 1));
                throw "Unexpected return code received.";

            }
        };
    };
}
