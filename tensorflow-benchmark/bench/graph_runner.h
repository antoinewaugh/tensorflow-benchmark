#pragma once

#include "tensorflow/core/public/session.h"
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <iostream>
#include <string>

using namespace tensorflow;

using graph_output_t = std::vector<Tensor>;
using graph_input_t = std::vector<std::pair<string, Tensor>>;

class graph_runner {

public:
    graph_runner(std::string path);
    ~graph_runner();
    graph_output_t predict(graph_input_t, std::string const&);

private:
    std::string path;
    Session* session;
    GraphDef graph_def;
};

graph_runner::graph_runner(std::string path):path(path) {

    auto status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return; // throw
    }

    status = ReadBinaryProto(Env::Default(), path, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return; // throw
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return; // throw
    }
}

// Copy Elision should make this efficient
// http://en.cppreference.com/w/cpp/language/copy_elision
graph_output_t graph_runner::predict(graph_input_t input_tensors, std::string const& input_labels) {

    Tensor output_tensor(DT_FLOAT, TensorShape());
    std::vector<Tensor> output_tensors = {output_tensor};

    auto status = session->Run(input_tensors, {input_labels}, {},
                          &output_tensors);

    if (!status.ok())
    {
        std::cout << "Error running graph: " +
                     status.ToString() << std::endl;
        // throw
    }

    return output_tensors;
}

graph_runner::~graph_runner() {
    session->Close();
}



