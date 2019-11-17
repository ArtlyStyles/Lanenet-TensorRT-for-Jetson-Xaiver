export PYTHONPATH=~/car/apps/lanenet-lane-detection
python3 tools/compare_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet_vgg.ckpt --image_path ./data/tusimple_test_image/0_512_256.jpg
