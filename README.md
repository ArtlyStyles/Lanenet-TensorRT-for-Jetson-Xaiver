# Lanenet-TensorRT-for-Jetson-Xaiver

I trained the Lanenet (https://github.com/MaybeShewill-CV/lanenet-lane-detection) on desktop and was able to run the trained mode with TensorRT on Nvidia Jetson AGX Xavier. The lanenet part is about 90ms for a 640x360 image on Xavier. 

How to convert and run the mode with TensorRT:

  1. You first need to convert the Tensorflow check point to UFF. There should be a lot of code out there to do it. When you do this, you need to select the output nodes for binary mask and seg mask. 
  
  2. Then use something similar to TensorRT sample code on Xavier under /usr/src/tensorrt/samples to run the UFF. 
  
However, when I worked on 2 above, I found I could not get the binary seg mask (which is the output of argmax and int64) to work properly in TensorRT. Maybe it was not supported by TensorRT. So I went back to 1, and selected the layer before the argmax, which is "Softmax", as output node for UFF. Then 2 worked fine. After that, I wrote my own image processing function same as argmax to process the output from TensorRT and get the final seg mask. 

Postprocessing is very slow: 

The DBscan based post processing is slow. It could be much faster if you write you own c++ code. You can find DBScan, curveFitting etc on github. The key is that when you use DBscan, do not put all pixels into it. You can select just some pixels, for example, skip every other pixel in the row, and skip every other row. Then you DBscan will be much faster. I was able to run the entire post-processing under a few ms. 

I was able to use lanenet on Xavier to drive a robocar autonomously along sidewalks without using any other algorithms https://twitter.com/SmallpixelCar/status/1297556145993707521





