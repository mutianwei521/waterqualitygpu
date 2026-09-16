# EPANET-dev/ — where the EPANET 2.3 source goes

The differentiable engine is not a reimplementation that replaces EPANET: it calls the real engine
through `ctypes`, reads hydraulics from the engine's own binary `.hyd` file, and checks every
gradient against the compiled solver. A local EPANET build is therefore required, and the example
networks `Net1.inp`, `Net2.inp` and `Net3.inp` are read from this checkout.

```bash
git clone https://github.com/OpenWaterAnalytics/EPANET.git EPANET-dev
cmake -S EPANET-dev -B EPANET-dev/build -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build EPANET-dev/build --config Release --target epanet2
export EPANET_LIB=$PWD/EPANET-dev/build/lib/libepanet2.so     # epanet2.dll on Windows
```

The reported runs used engine version 20306 (EPANET 2.3). `diffwq/epanet_api.py` resolves the
library from `$EPANET_LIB` first; see its `default_library_path()` for the fallbacks.

On Windows we built with MSVC in `RelWithDebInfo` rather than `Release`, because the project's
`Release` flags set `/GL` without an explicit speed option, which would have made the timing
baseline unrepresentative. See `research/water_quality_2026_09_12/REPRODUCE.md`.
