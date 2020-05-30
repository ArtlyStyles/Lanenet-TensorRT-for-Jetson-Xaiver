# Lanenet-TensorRT-for-Jetson-Xaiver

You first need to convert the Tensorflow check point to UFF. 

Then use something similar to TensorRT sample code to run the UFF. I found, the binary seg mask, which is the output of argmax and int64, could not work properlly in TensorRT, so I just the layer before it, "Softmax" as output node for UFF. 
After get the output from inference, you can implement something similar to argmax. 

