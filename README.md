# Lanenet-TensorRT-for-Jetson-Xaiver

I trained the LannYou first need to convert the Tensorflow check point to UFF. There should be a lot of code out there to do it. 

Then use something similar to TensorRT sample code under /usr/src/tensorrt/samples to run the UFF. 

I could not get the binary seg mask (which is the output of argmax and int64) to work properly in TensorRT, so I just used the layer before the argmax, which is "Softmax", as output node for UFF. After get the output from inference, I implemented something similar to argmax.



