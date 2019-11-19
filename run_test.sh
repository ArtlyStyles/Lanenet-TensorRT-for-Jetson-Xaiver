export PYTHONPATH=~/car/apps/lanenet-lane-detection
python3 tools/test_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt --image_path ./data/tusimple_test_image/0.jpg
