#pragma once

#include "http/httplib.h"

#include "pipeline.h"

class RWKVServer
{
public:
    RWKVServer(RWKVPipeline &pipeline);

    void start(std::string ip, uint32_t port);

private:
    httplib::Server svr_;
    RWKVPipeline pipeline_;
};