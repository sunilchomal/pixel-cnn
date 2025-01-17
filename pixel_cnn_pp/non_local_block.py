"""
    source: https://github.com/lucasb-eyer/nonlocal-tf
    TensorFlow (no Keras) implementation of the building blocks described in 
    Non-local Neural Networks by Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    https://arxiv.org/abs/1711.07971

"""
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.

    Note that

        net = conv2d_same(inputs, num_outputs, 3, stride=stride)

    is equivalent to

        net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
        net = subsample(net, factor=stride)

    whereas

        net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

    Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

    Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    slim = tf.contrib.slim
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                            padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                            rate=rate, padding='VALID', scope=scope)

@add_arg_scope
def nonlocal_dot(net, depth, embed=True, softmax=False, maxpool=2, scope=None):
    """ Implementation of the non-local block in its various forms.
    See "Non-local Neural Networks" by
    Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He
    https://arxiv.org/pdf/1711.07971.pdf

    Args:
    - `net`: The symbolic input into the block, a (B,H,W,C) Tensor.
    - `depth`: The number of channels in which to execute the non-local operation.
    - `embed`: Whether or not use the "embedded version" as in Sec.3.2
    - `softmax`: Whether or not to use the softmax operation which makes it
                equivalent to soft-attention.
    - `maxpool`: How large of a max-pooling (Sec.3.3) to use to help reduce
                the computational burden. Default is 2, use `False` for none.
    - `scope`: An optional scope for all created variables.

    Returns:
    The symbolic output of the non-local block operation.

    Note:
    The final BatchNorm's gamma is initialized to zero, so as to make this a
    no-op (skip) at initialization, as described in Sec.4.1.
    """
    slim = tf.contrib.slim
    with tf.variable_scope(scope, 'nonlocal', values=[net]) as sc:
        with slim.arg_scope([slim.conv2d], normalizer_fn=None):
            if embed:
                a = conv2d_same(net, depth, 1, stride=1, scope='embA')
                b = conv2d_same(net, depth, 1, stride=1, scope='embB')
            else:
                a, b = net, net
            g_orig = g = conv2d_same(net, depth, 1, stride=1, scope='g')
        if maxpool is not False and maxpool > 1:
            b = slim.max_pool2d(b, [maxpool, maxpool], stride=maxpool, scope='pool')
            g = slim.max_pool2d(g, [maxpool, maxpool], stride=maxpool, scope='pool')

    # Flatten from (B,H,W,C) to (B,HW,C) or similar
    a_flat = tf.reshape(a, [tf.shape(a)[0], -1, tf.shape(a)[-1]])
    b_flat = tf.reshape(b, [tf.shape(b)[0], -1, tf.shape(b)[-1]])
    g_flat = tf.reshape(g, [tf.shape(g)[0], -1, tf.shape(g)[-1]])
    a_flat.set_shape([a.shape[0], a.shape[1] * a.shape[2] if None not in a.shape[1:3] else None, a.shape[-1]])
    b_flat.set_shape([b.shape[0], b.shape[1] * b.shape[2] if None not in b.shape[1:3] else None, b.shape[-1]])
    g_flat.set_shape([g.shape[0], g.shape[1] * g.shape[2] if None not in g.shape[1:3] else None, g.shape[-1]])
    # Compute f(a, b) -> (B,HW,HW)
    f = tf.matmul(a_flat, tf.transpose(b_flat, [0, 2, 1]))
    if softmax:
        f = tf.nn.softmax(f)
    else:
        f = f / tf.cast(tf.shape(f)[-1], tf.float32)
    # Compute f * g ("self-attention") -> (B,HW,C)
    fg = tf.matmul(f, g_flat)
    # Expand and fix the static shapes TF lost track of.
    fg = tf.reshape(fg, tf.shape(g_orig))
    # fg.set_shape(g.shape)  # NOTE: This actually appears unnecessary.

    # Go back up to the original depth, add residually, zero-init.
    #with slim.arg_scope([slim.conv2d],
    #                    weights_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.batch_norm], param_initializers={'gamma': tf.zeros_initializer()}):
        fg = conv2d_same(fg, net.shape[-1], 1, stride=1, scope='fgup')
    net = net + fg

    return slim.utils.collect_named_outputs(None, sc.name, net)