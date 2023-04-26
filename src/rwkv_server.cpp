#include "rwkv_server.h"

RWKVServer::RWKVServer(RWKVPipeline &pipeline):pipeline_(pipeline)
{
    svr_.Get("/api/chat", 
        [this](const httplib::Request &req, httplib::Response &res){  
            std::string context = req.get_param_value("text");
            float temperature = std::stof(req.get_param_value("temperature"));
            float top_p = std::stof(req.get_param_value("topP"));
            int token_count = std::stoi(req.get_param_value("tokenCount"));
            float fresence_penalty = std::stof(req.get_param_value("presencePenalty"));
            float count_penalty = std::stof(req.get_param_value("countPenalty"));
            // std::cout<<"request:"<<context<<temperature<<top_p<<token_count<<fresence_penalty<<count_penalty<<"\n";
            std::string out_str;
            try
            {
                out_str = this->pipeline_.generate(context, temperature, top_p, token_count, fresence_penalty, count_penalty, std::vector<uint32_t>{}, std::vector<uint32_t>{0});
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
            res.set_content(out_str,"text/plain"); 
        }
    );
    svr_.Get("/api/write",
        [this](const httplib::Request &req, httplib::Response &res){  
            std::string context = req.get_param_value("text");
            float temperature = std::stof(req.get_param_value("temperature"));
            float top_p = std::stof(req.get_param_value("topP"));
            int token_count = std::stoi(req.get_param_value("tokenCount"));
            float fresence_penalty = std::stof(req.get_param_value("presencePenalty"));
            float count_penalty = std::stof(req.get_param_value("countPenalty"));
            std::string out_str;
            try
            {
                out_str = this->pipeline_.generate(context, temperature, top_p, token_count, fresence_penalty, count_penalty, std::vector<uint32_t>{}, std::vector<uint32_t>{0});
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
            
            res.set_content(out_str,"text/plain");
        }
    );
}

void RWKVServer::start(std::string ip, uint32_t port)
{
    std::cout << "start listening" << std::endl;
    svr_.listen(ip, port);
    std::cout << "end";
}