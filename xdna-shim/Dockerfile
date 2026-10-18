# Builds libvoxtype_xrt.so against Debian trixie-backports' XRT and collects
# the XRT runtime libraries it needs, so a host without XRT can run the NPU
# encoder from one directory:
#
#   docker build -f xdna-shim/Dockerfile --output type=local,dest=out xdna-shim
#
# XRT finds its own core and driver libraries relative to libxrt_coreutil, at
# <prefix>/lib/x86_64-linux-gnu, so the output keeps Debian's layout and the
# shim (RUNPATH=$ORIGIN) sits next to them in that directory.
FROM debian:13 AS build
RUN echo "deb http://deb.debian.org/debian trixie-backports main" \
      > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++ uuid-dev \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      -t trixie-backports libxrt-dev libxrt-npu2 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY vxrt.h vxrt.cpp smoke.c ./
RUN mkdir -p /out/lib/x86_64-linux-gnu \
 && g++ -std=c++17 -O2 -fPIC -shared -fvisibility=default \
      -Wl,-soname,libvoxtype_xrt.so -Wl,-rpath,'$ORIGIN' \
      -o /out/lib/x86_64-linux-gnu/libvoxtype_xrt.so vxrt.cpp \
      -lxrt_coreutil \
 && gcc -O2 -o /out/lib/x86_64-linux-gnu/vxrt-smoke smoke.c -L/out/lib/x86_64-linux-gnu -lvoxtype_xrt \
      -Wl,-rpath,'$ORIGIN' \
 && for l in libxrt_coreutil libxrt_core libxrt_driver_xdna; do \
      cp -a /usr/lib/x86_64-linux-gnu/$l.so.2* /out/lib/x86_64-linux-gnu/; \
    done \
 && dpkg-query -W -f='${Package} ${Version}\n' libxrt-npu2 libxrt2 > /out/XRT_VERSION

FROM scratch
COPY --from=build /out/ /
