#include "tensorflow/core/public/session.h"
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <iostream>
#include <string>

using namespace tensorflow;

int main(int argc, char *argv[])
{
/*        if (argc <= 1) {
		std::cout << "Usage: tensorflow-benchmark /path/to/*.pb" << '\n';
        }*/

//	std::string pathToGraph(argv[1]);
	std::string pathToGraph("../graph/frozen_model.pb");

	// Initialize a tensorflow session
	Session *session;
	auto status = NewSession(SessionOptions(), &session);
	if (!status.ok())
	{
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Read in the protobuf exported graph
	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Setup inputs and outputs:
	// These input tensor nodes have been previously named in the graph
	// as x,y therefore we maintain consistency in c++ variable naming
	//	Tensor x(DT_FLOAT, TensorShape()), y(DT_FLOAT, TensorShape());

	Tensor mariana_feature(DT_FLOAT, {1,12});
	auto t_matrix = mariana_feature.matrix<float>();
	t_matrix(0,0) = 0.35075195;
	t_matrix(0,1) = 0.67798291;
	t_matrix(0,2) = 0.35005933;
	t_matrix(0,3) = 0.21196908;
	t_matrix(0,4) = 0.52058227;
	t_matrix(0,5) = 0.28996623;
	t_matrix(0,6) = 0.37157229;
	t_matrix(0,7) = 0.40275239;
	t_matrix(0,8) = 0.4685295;
	t_matrix(0,9) = 0.63283459;
	t_matrix(0,10) = 0.52569881;
	t_matrix(0,11) = 0.62036159;

	std::cout << t_matrix << std::endl;

	std::vector<std::pair<string, Tensor>> input_tensors = {{"main_input", mariana_feature}};


	Tensor mariana_output(DT_FLOAT, TensorShape());
	std::vector<Tensor> output_tensors = {
			mariana_output}; // need to explicitly set shape, type?


	//warm up
	status = session->Run(input_tensors, {"main_output/Softmax"}, {},
	                      &output_tensors);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	for(int i=0;i<10000;i++)
	{
		status = session->Run(input_tensors, {"main_output/Softmax"}, {},
	                      &output_tensors);
	}

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;

	if (!status.ok())
	{
		std::cout << "Error running graph: " +
		             status.ToString() << std::endl;
	}

	std::cout << std::endl;
	std::cout << "***** Result ***** " << std::endl;
	std::cout << std::endl;

	auto output_c = output_tensors[0].matrix<float>();
	std::cout << output_c << std::endl;

	session->Close();
	return 0;
}


