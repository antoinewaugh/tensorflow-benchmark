#include "graph_runner.h"
#include "benchmark/benchmark.h"


static void BM_GraphCall(benchmark::State& state) {

    graph_runner runner("../../graph/frozen_model.pb");
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

    std::vector<std::pair<string, Tensor>> input_tensors = {{"main_input", mariana_feature}};
    auto input_label = "main_output/Softmax";

    for (auto _ : state) {
          auto val = runner.predict(input_tensors, input_label);
      }
}

BENCHMARK(BM_GraphCall);
BENCHMARK_MAIN();
