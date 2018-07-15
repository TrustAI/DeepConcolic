import matplotlib.pyplot as plt


def is_conv_layer(layer):
    return (layer.name.find('conv') >= 0)


def is_dense_layer(layer):
    return (layer.name.find('dense') >= 0)


def is_activation_layer(layer):
    return (layer.name.find('activation') >= 0)


def is_maxpooling_layer(layer):
    return (layer.name.find('max_pooling') >= 0)


def is_flatten_layer(layer):
    return (layer.name.find('flatten') >= 0)


def is_dropout_layer(layer):
    return False  ## we do not allow dropout


### given an input image, to evaluate activations
def eval(layer_functions, im):
    activations = []
    for l in range(0, len(layer_functions)):
        if l == 0:
            activations.append(layer_functions[l]([[im]])[0])
        else:
            activations.append(layer_functions[l]([activations[l - 1]])[0])
    return activations


def eval_batch(layer_functions, ims):
    activations = []
    for l in range(0, len(layer_functions)):
        if l == 0:
            activations.append(layer_functions[l]([ims])[0])
        else:
            activations.append(layer_functions[l]([activations[l - 1]])[0])
    return activations


def show_adversarial_examples(imgs, ys, name):
    for i in range(0, 2):
        plt.subplot(1, 2, 1 + i)
        print 'imgs[i].shape is ', imgs[i].shape
        plt.imshow(imgs[i].reshape([28, 28]), cmap=plt.get_cmap('gray'))
        plt.title("label: " + str(ys[i]))
        plt.savefig(name, bbox_inches='tight')
