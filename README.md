## 轻量级人脸检测+关键点检测
ncnn部署轻量级人脸检测模型 rhttps://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB，和insightface人脸关键点检测模型 https://github.com/deepinsight/insightface。
人脸检测的输入分辨率为320x240，人脸检测+关键点在rk3566设备单线程CPU能达15-18FPS。rk3588单线程能达30FPS以上。（占用低、速度快，Linzaer大佬的模型非常好用）

This is a sample ncnn android project, it depends on ncnn library and opencv

https://github.com/Tencent/ncnn

https://github.com/nihui/opencv-mobile


## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
https://github.com/nihui/opencv-mobile

* Download opencv-mobile-XYZ-android.zip
* Extract opencv-mobile-XYZ-android.zip into **app/src/main/jni** and change the **OpenCV_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step3
* Open this project with Android Studio, build it and enjoy!

### screenshot
!()[face.png]
!()[cpu.png]

## reference
https://github.com/nihui/ncnn-android-mobilenetssd
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
https://github.com/deepinsight/insightface

