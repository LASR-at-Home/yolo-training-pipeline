{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  }
, cudaPkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  }
, pythonPackages ? pkgs.python310Packages
, cudaPythonPackages ? cudaPkgs.python310Packages
}:

pkgs.mkShell rec {
  name = "yoloEnv";
  venvDir = "./.venv";

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.gcc-unwrapped
    pkgs.zlib
    pkgs.glib
    pkgs.libGL
  ];

  buildInputs = [
    # Tools
    pkgs.git
    pkgs.ffmpeg-full

    # CUDA
    cudaPkgs.cudatoolkit

    # Python
    pythonPackages.pip
    pythonPackages.python
    pythonPackages.virtualenv

    # Python Packages
    pythonPackages.numpy
    pythonPackages.opencv4
    pythonPackages.matplotlib
    cudaPythonPackages.pytorch-bin
    cudaPythonPackages.torchvision-bin
  ];

  # Switch to virtualenv when opening shell
  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)

    __activate_venv() {
      # Under some circumstances it might be necessary to add your virtual
      # environment to PYTHONPATH, which you can do here too;
      # PYTHONPATH=$PWD/${venvDir}/${pythonPackages.python.sitePackages}/:$PYTHONPATH

      # Save current PATH as activating virtualenv seems to clear it
      __OLD_PATH=$PATH

      source "${venvDir}/bin/activate"
      export PATH=$PATH:$__OLD_PATH
    }

    if [ -d "${venvDir}" ]; then
      echo "Skipping venv creation, '${venvDir}' already exists"
      __activate_venv
    else
      echo "Creating new venv environment in path: '${venvDir}'"
      # Note that the module venv was only introduced in python 3, so for 2.7
      # this needs to be replaced with a call to virtualenv
      ${pythonPackages.python.interpreter} -m venv "${venvDir}"

      # Switch to virtualenv
      __activate_venv

      # Install required packages
      pip install -r requirements.txt
    fi

    export CUDA_PATH=${pkgs.cudatoolkit}
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}
