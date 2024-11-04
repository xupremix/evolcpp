Installation:

```bash
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
sudo cp -r libtorch /usr/lib/
rm -rf libtorch
rm libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/lib/libtorch/ -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
cd ..
make
```
