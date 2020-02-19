# gesture-driven-presentations

All of this is a WIP and subject to a lot of modification :).

## Getting started (on Windows!)

So, first:
* install Python: https://www.python.org/downloads/ . It should also come with pip.
* TIP: you could use a virtual environment. I use virtualenv and virtualenvwrapper, there's a nice tutorial here: http://www.tumblingprogrammer.com/setting-up-windows-for-python-development-python-2-python-3-git-bash-terminal-and-virtual-environments/#installing-virtual-environment-and-virtual-environment-wrapper
* install the requirements: `pip install -r requirements.txt`
* install OpenCV: I personally found it easiest to install with conda: `conda install opencv` should work.
* (best option we have so far) install PointerFocus (for some cursor functionalities that we will use): http://www.pointerfocus.com/

To see that everything works so far, try running `src/test_wrappers.py` and perhaps `src/recognize_classical_cv.py` :).

Next up: you also need to build OpenPose!

## Build OpenPose!

Follow the steps below. Important notes that you will need during the installation (which are not mentioned in the steps in the link)
* enable BUILD_PYTHON in CMAKE GUI
* in Visual Studio, make sure you compile the whole solution: right-click on the OpenPose project solution and click Build Solution

https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md

OK, after you've done that you need to copy some OpenPose binaries to our project folder.
In the root of the repository create an "openpose" folder. In that folder, copy the following folders:
* "models" folder from the root of your openpose repository
* "bin" folder from the build folder in your openpose repository.

Next, in your newly created "openpose" folder, create a "lib" folder. In that folder, copy the following folders:
* "python" folder from the build folder in your openpose repository
* "x64" folder from the build folder in your openpose repository.

So, in the end you should have:
```
gesture-driven-presentations
    > README.md (... and other things...)
    > openpose
        > bin
        > lib
            > python
            > x64
        > models
```

Now, you should be able to run `src/recognize_openpose.py`! :)
