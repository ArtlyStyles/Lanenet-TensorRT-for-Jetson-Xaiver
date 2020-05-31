# Lanenet-TensorRT-for-Jetson-Xaiver

You first need to convert the Tensorflow check point to UFF. 

Then use something similar to TensorRT sample code under /usr/src/tensorrt/samples to run the UFF. I could not get the binary seg mask, which is the output of argmax and int64, to work propery in TensorRT, so I just use the layer before it, "Softmax", as output node for UFF. After get the output from inference, I implemented something similar to argmax.



