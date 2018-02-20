#include "benchmark/benchmark.h"

static void BM_Increment(benchmark::State& state) {
  int i = 0;
  while (state.KeepRunning()) {
	  i++;
  }
}
BENCHMARK(BM_Increment);
BENCHMARK_MAIN();
