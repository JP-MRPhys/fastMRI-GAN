import  tensorflow as tf
import numpy as np


from models.utils.Layers import linear, upsampling_2D


def generator(z):
    """
    :param z: random array input dimension (batch_size, z_dim)
    :return: image (Gz)
    """

    batch_size=10

    z_= linear(z, 64 * 4 * 4 * 8, 'g_h0_lin')

    z_resize = tf.reshape(z_, [-1, 4, 4, 64 * 8])  # add a-relu
    z_resize = tf.nn.relu(z_resize)
    # up_1 = upsampling(z_resize, [self.batch_size, 8, 8], 512, 1024, 2, name='g_up2')
    up_2 = upsampling_2D(z_resize, [batch_size, 8, 8], 256, 512, 2, name='g_up3', last_layer=False)
    up_3 = upsampling_2D(up_2, [batch_size, 16, 16], 128, 256, 2, name='g_up4', last_layer=False)
    up_4 = upsampling_2D(up_3, [batch_size, 32, 32], 32, 128, 2, name='g_up5', last_layer=False)
    up_5 = upsampling_2D(up_4, [batch_size, 64, 64], 16, 32, 2, name='g_up6', last_layer=False)
    up_6 = upsampling_2D(up_5, [batch_size, 128, 128], 32, 16, 2, name='g_up7', last_layer=False)
    up_7 = upsampling_2D(up_6, [batch_size, 256, 256], 1, 32, 2, name='g_up8', last_layer=True)

    print("Completed creating generator with last layer shape of")
    print(up_7)

    return tf.nn.tanh(up_7)


if __name__ == '__main__':


    z=tf.placeholder(tf.float32, shape=[10,100])
    Gz=generator(z)

    for i in range (100):
        z_samples = np.random.uniform(-1, 1, size=(10)).astype(np.float32)
        print(z_samples)



