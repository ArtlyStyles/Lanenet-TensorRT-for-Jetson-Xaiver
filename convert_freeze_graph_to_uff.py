import uff

frozen_filename ='./lanenet-model-segment-native.pb'
output_node_names = ['lanenet_model/vgg_backend/binary_seg/ArgMax']
#output_node_names = ['lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/Conv2D','lanenet_model/vgg_backend/binary_seg/ArgMax']
#output_node_names = ['lanenet_model/vgg_backend/instance_seg/pix_embedding_conv/Conv2D']
output_uff_filename = './lanenet_segment_model.uff'

#uff_mode = uff.from_tensorflow_frozen_model(frozen_file=frozen_filename, output_nodes=output_node_names, output_filename=output_uff_filename, text=False)
uff_mode = uff.from_tensorflow_frozen_model(frozen_filename, output_nodes=output_node_names, output_filename=output_uff_filename, text=False)
