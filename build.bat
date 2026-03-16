@echo off
setlocal

REM Go to this project
cd /d C:\Users\11032\repos\openvino_costum_cnn_cpp

echo Cleaning build folder...
rmdir /S /Q build 2>nul
mkdir build
cd build

echo Loading OpenVINO environment...
call "C:\Intel\openvino_2026\setupvars.bat"

echo Setting OpenCV...
set OpenCV_DIR=C:\libs\opencv\build
set PATH=C:\libs\opencv\build\x64\vc16\bin;%PATH%

echo Running CMake...
cmake -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DOpenCV_ARCH=x64 ^
  -DOpenCV_RUNTIME=vc16 ^
  -DOpenCV_STATIC=OFF ^
  ..

echo Building...
cmake --build . --config Release

echo Done.
pause