
cd $(dirname "$0")

dst_dir=$(pip show tensorrt_llm | grep "Location:" | cut -d' ' -f2)/tensorrt_llm

echo "Copying built shared libraries to $dst_dir"

cd cpp/build_Debug

# # cmake ..
cmake --build . -j$(nproc) --target build_wheel_targets bindings

find . -name "libtensorrt_llm.so" -exec cp -vf {} "$dst_dir/libs" \;
find . -name "libnvinfer_plugin_tensorrt_llm.so" -exec cp -vf {} "$dst_dir/libs" \;
find . -name "bindings*.so" -exec cp -vf {} "$dst_dir" \;
