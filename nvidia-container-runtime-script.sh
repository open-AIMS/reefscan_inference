curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  tee /etc/apt/sources.list.d/nvidia-container-runtime.list
apt-get update -y