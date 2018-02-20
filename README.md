# tensorflow-benchmark
benchmarking tool for hitting TF graph.

## install tensorflow-cc shared lib for cmake
```
git clone https://github.com/FloopCZ/tensorflow_cc.git
cd tensorflow_cc
mkdir build && cd $_
cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON ..
make && sudo make install
```

## install tensorflow-benchmark
```
git clone https://github.com/antoinewaugh/tensorflow-benchmark.git
cd tensorflow-benchmark
mkdir build && cd $_
cmake .. && make
```
