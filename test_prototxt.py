import caffe
import numpy as np

def wsweights(weights, gain):
  mean = np.mean(weights, axis=(0, 1, 2))
  var = np.var(weights, axis=(0, 1, 2))
  fan_in = np.prod(weights.shape[:-1])
  scale = gain / np.sqrt(np.maximum(var * fan_in, 1e-4))
  shift = mean * scale
  return weights * scale - shift

def load_block(net, haiku_params, block_number):
  if block_number == 0:
    name_block = "NFNet.nf_block"
  else:
    name_block = "NFNet.nf_block_{}".format(block_number)
  # projection
  if "{}.conv_shortcut.w".format(name_block) in haiku_params.keys():
    conv_shortcut_w = wsweights(haiku_params["{}.conv_shortcut.w".format(name_block)], haiku_params["{}.conv_shortcut.gain".format(name_block)])
    conv_shortcut_w = np.transpose(conv_shortcut_w, (3, 2, 0, 1))
    np.copyto(net.params["block{}_conv_shortcut".format(block_number)][0].data, conv_shortcut_w)
    np.copyto(net.params["block{}_conv_shortcut".format(block_number)][1].data, haiku_params["{}.conv_shortcut.bias".format(name_block)])
  # conv 0
  conv0_w = wsweights(haiku_params["{}.conv0.w".format(name_block)], haiku_params["{}.conv0.gain".format(name_block)])
  conv0_w = np.transpose(conv0_w, (3, 2, 0, 1))
  np.copyto(net.params["block{}_conv0".format(block_number)][0].data, conv0_w)
  np.copyto(net.params["block{}_conv0".format(block_number)][1].data, haiku_params["{}.conv0.bias".format(name_block)])
  # conv 1
  conv1_w = wsweights(haiku_params["{}.conv1.w".format(name_block)], haiku_params["{}.conv1.gain".format(name_block)])
  conv1_w = np.transpose(conv1_w, (3, 2, 0, 1))
  np.copyto(net.params["block{}_conv1".format(block_number)][0].data, conv1_w)
  np.copyto(net.params["block{}_conv1".format(block_number)][1].data, haiku_params["{}.conv1.bias".format(name_block)])
  # conv 1b
  if "{}.conv1b.w".format(name_block) in haiku_params.keys():
    conv1b_w = wsweights(haiku_params["{}.conv1b.w".format(name_block)], haiku_params["{}.conv1b.gain".format(name_block)])
    conv1b_w = np.transpose(conv1b_w, (3, 2, 0, 1))
    np.copyto(net.params["block{}_conv1b".format(block_number)][0].data, conv1b_w)
    np.copyto(net.params["block{}_conv1b".format(block_number)][1].data, haiku_params["{}.conv1b.bias".format(name_block)])
  np.copyto(net.params["block{}_conv1".format(block_number)][1].data, haiku_params["{}.conv1.bias".format(name_block)])
  # conv 2
  conv2_w = wsweights(haiku_params["{}.conv2.w".format(name_block)], haiku_params["{}.conv2.gain".format(name_block)])
  conv2_w = np.transpose(conv2_w, (3, 2, 0, 1))
  np.copyto(net.params["block{}_conv2".format(block_number)][0].data, conv2_w)
  np.copyto(net.params["block{}_conv2".format(block_number)][1].data, haiku_params["{}.conv2.bias".format(name_block)])
  # squeeze and excite
  np.copyto(net.params["block{}_se_fc1".format(block_number)][0].data, np.transpose(haiku_params["{}.squeeze_excite.linear.w".format(name_block)]))
  np.copyto(net.params["block{}_se_fc1".format(block_number)][1].data, haiku_params["{}.squeeze_excite.linear.b".format(name_block)])
  np.copyto(net.params["block{}_se_fc2".format(block_number)][0].data, np.transpose(haiku_params["{}.squeeze_excite.linear_1.w".format(name_block)]))
  np.copyto(net.params["block{}_se_fc2".format(block_number)][1].data, haiku_params["{}.squeeze_excite.linear_1.b".format(name_block)])
  print("Done loading {} ".format(name_block))

caffe.set_mode_gpu()

net = caffe.Net("./nfnet-F0.prototxt", caffe.TEST)

#net.forward()
#net.backward()

haiku_params = dict(np.load("./F0_weights.npy", allow_pickle=True).item())

stem_conv0_w = wsweights(haiku_params["NFNet.stem_conv0.w"], haiku_params["NFNet.stem_conv0.gain"])
stem_conv0_w = np.transpose(stem_conv0_w, (3, 2, 0, 1))
np.copyto(net.params['stem_conv0'][0].data, stem_conv0_w)
np.copyto(net.params['stem_conv0'][1].data, haiku_params["NFNet.stem_conv0.bias"])

stem_conv1_w = wsweights(haiku_params["NFNet.stem_conv1.w"], haiku_params["NFNet.stem_conv1.gain"])
stem_conv1_w = np.transpose(stem_conv1_w, (3, 2, 0, 1))
np.copyto(net.params['stem_conv1'][0].data, stem_conv1_w)
np.copyto(net.params['stem_conv1'][1].data, haiku_params["NFNet.stem_conv1.bias"])

stem_conv2_w = wsweights(haiku_params["NFNet.stem_conv2.w"], haiku_params["NFNet.stem_conv2.gain"])
stem_conv2_w = np.transpose(stem_conv2_w, (3, 2, 0, 1))
np.copyto(net.params['stem_conv2'][0].data, stem_conv2_w)
np.copyto(net.params['stem_conv2'][1].data, haiku_params["NFNet.stem_conv2.bias"])

stem_conv3_w = wsweights(haiku_params["NFNet.stem_conv3.w"], haiku_params["NFNet.stem_conv3.gain"])
stem_conv3_w = np.transpose(stem_conv3_w, (3, 2, 0, 1))
np.copyto(net.params['stem_conv3'][0].data, stem_conv3_w)
np.copyto(net.params['stem_conv3'][1].data, haiku_params["NFNet.stem_conv3.bias"])

i = 11
for i in range(i+1):
  load_block(net, haiku_params, i)

final_conv_w = wsweights(haiku_params["NFNet.final_conv.w"], haiku_params["NFNet.final_conv.gain"])
final_conv_w = np.transpose(final_conv_w, (3, 2, 0, 1))
np.copyto(net.params['final_conv'][0].data, final_conv_w)
np.copyto(net.params['final_conv'][1].data, haiku_params["NFNet.final_conv.bias"])

np.copyto(net.params["fc"][0].data, np.transpose(haiku_params["NFNet.linear.w"]))
np.copyto(net.params["fc"][1].data, haiku_params["NFNet.linear.b"])


net.blobs['data'].data.fill(1)
net.forward();
#print(net.blobs['block{}_add'.format(i)].data[0, :, 4, 4])
print(net.blobs['fc'].data)
