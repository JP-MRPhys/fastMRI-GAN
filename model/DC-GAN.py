"""
DC-GAN to leaning probablity distribution for MRI image objects

Features:

1. Generator (U-NET) with strided convolutiono (i.e. up-sampling)
   f. No Pooling layer, Batch normalization, relu activation for each layer except for the output layer activation (tanh)


2. Discriminator
   a. replaced pooling layers by strided convolutions, batch norm except the discriminator input layer
   b. No fully connected layer at output (replaced with Flatten output)
   c. leaky-relu activation

3. Adversarial Training.

4. Adding Perceptual (VGG) loss and pixelwise loss to reduce "halluincation" in the generator outputs, loss function as-per DAGAN, Yang et al, IEEE, 2018

5. For employing Conditional -DC-GAN (changes required as per comments in the code)



"""

def fft_abs_for_map_fn(x):
    x = (x + 1.) / 2.
    x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs



import pathlib
#import nibabel as nib
import random
from model.utils.Layers import *
from model.fastmri_data import  get_random_accelerations, get_training_pair

# Just disables the warning, doesn't enable AVX/FMA


class DCGAN:
    def __init__(self, vggdir, name):
        # network parameters
        self.vggdir = vggdir
        self.training_datadir='/media/jehill/Data/ML_data/fastmri/singlecoil/train/singlecoil_train/'
        #self.labeldir=labeldir
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.num_epochs = 1000
        self.display_step = 20
        self.global_step = 0
        self.w = 128  # x
        self.h = 128  # y
        self.z_dim = 100
        self.w2 = self.w / 2  #
        self.h2 = self.h / 2  #
        self.d = 1  # z or channels
        self.X_train = tf.placeholder(tf.float32, [None, None, None, self.d], name='X_train')
        self.X_conditioning = tf.placeholder(tf.float32, [None, None, None, self.d+1], name='X_train_conditioning')
        self.batch_size = 30;
        self.num_classes = 10  # anging number of features to 5
        self.g_gamma = 0.0025  # 0.025weight for perceptual loss
        self.g_alpha = 0.1  # 0.1weight for pixel loss
        self.g_beta = 0.1  # 0.1weight for frequency loss
        self.g_adv = 1  # weight for frequency loss

        self.training_dir = []
        self.labels_dir = []
        self.images_tr = None


        # now create the network
        self.keep_prob = 0.8  # that the drop
        self.drop_out = self.keep_prob

        # Initialize Network weights
        self.initializer = tf.truncated_normal_initializer(stddev=0.2)

        # Input for Generator
        # self.z_in = tf.placeholder(tf.float32, [None, self.w, self.h, self.d], name='z_in')
        print(str(self.w2))
        print(str(self.h2))

        # Inputs for Discriminator and conditioning mask

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # self.input_image = tf.placeholder(shape=[None, self.w, self.h, self.d], dtype=tf.float32, name='input_image')
        # self.conditioning_input = tf.placeholder(shape=[None, self.w, self.h, self.d], dtype=tf.float32,name="conditioning_input")  # must be of the same dimension as the input_image

        # self.X_train=distort_img(self.X_train)

        self.input_image = tf.image.resize_images(self.X_train, [np.int(self.w), np.int(self.h)])
        self.conditioning_input = tf.image.resize_images(self.X_conditioning, [np.int(self.w), np.int(self.h)])
        self.input_image_resize = tf.image.resize_images(self.input_image, [np.int(self.w2), np.int(self.h2)])
        self.input_image_244 = tf.image.resize_images(self.input_image, [244, 244])  # resize the the input image to VGG

        # self.conditioning_input2 = tf.image.resize_images(self.conditioning_input, tf.constant([self.w/2])) #generator images are half size so we resize our images
        self.conditioning_input_resize = tf.image.resize_images(self.conditioning_input, [np.int(self.w2), np.int(self.h2)])  # these needs to fixed using variable

        # Creating Images for ranom vectors (replaced with a U-NET)
        # self.generator_logits = self.generator(self.input_image, self.conditioning_input)
        # self.Gz = tf.reduce_mean(self.generator_logits, 3, keepdims=True, name='generator_output')

        self.Gz = self.generator_2(self.z)
        self.print_shape(self.Gz)
        self.Gz_244 = tf.image.resize_images(self.Gz, [244, 244])
        # self.Gz = tf.image.resize_images(self.Gz, [self.w, self.h])

        # Probabilities for real images
        # self.Dx, self.Dx_logits = self.discriminator(self.input_image, self.conditioning_input)
        self.Dx, self.Dx_logits = self.discriminator_2(self.input_image)

        # Probabilities for generator images
        print("Discriminator Shape 2:")
        # self.Dz, self.Dz_logits = self.discriminator(self.Gz, self.conditioning_input, reuse=True)
        self.Dz, self.Dz_logits = self.discriminator_2(self.Gz, reuse=True)

        # Adversarial training using cross entropy for G and D loss, plus additional losses
        # Discriminator loss

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_logits, labels=tf.ones_like(self.Dx)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.zeros_like(self.Dz)))

        self.d_loss = self.d_loss_fake + self.d_loss_real

        # Generator loss (adversarial)
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.ones_like(self.Dz)))




        # get the gradients for the generator and discriminator
        self.tvars = tf.trainable_variables()
        self.d_gradients = [var for var in self.tvars if 'd_' in var.name]
        self.g_gradients = [var for var in self.tvars if 'g_' in var.name]



        # Use the Adam Optimizers for discriminator and generator
        # LR = self.learning_rate
        # BTA = 0.5

        self.OptimizerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.d_loss,
                                                                                                       var_list=self.d_gradients)
        self.OptimizerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.g_loss,
                                                                                                       var_list=self.g_gradients)

        # summary and writer for tensorboard visulization

        # tf.summary.image("Segmentation", tf.to_float(self.segmented_image))
        tf.summary.image("Generator fake output", self.Gz)
        #tf.summary.image("Small image", self.Gz_small)
        tf.summary.image("Input image", self.input_image)
        #tf.summary.image("Mask image", self.conditioning_input_resize)

        tf.summary.histogram("Descriminator logits (Real)", self.Dx_logits)
        tf.summary.histogram("Descriminator logits (Fake)", self.Dz_logits)

        tf.summary.scalar("Discriminator loss real", self.d_loss_real)
        tf.summary.scalar("Generator loss fake", self.d_loss_fake)
        tf.summary.scalar("Total Discriminator loss", self.d_loss)
        tf.summary.scalar("Generator loss", self.g_loss)

        self.merged_summary = tf.summary.merge_all()

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.logdir = './' + name  # if not exist create logdir
        self.model_dir = self.logdir + 'final_model'
        self.model_name = name
        self.model_name2 = name

        print("Completed creating the tensor-flow model")

    # we employ y for conditioning by concat it with the input
    # def discriminator(self, image, conditioning_layer, reuse=False):
    def discriminator(self, image, reuse=False):

        PADDING = "SAME"
        STRIDE = 2  # [2, 2]

        # we employ y for conditioning by concat it with the input
        # input = tf.concat((image, conditioning_layer), 3)
        input = image

        # Conv Layer 1, No batch normalization, leaky relu activation
        d1_conv = slim.convolution2d(input, 16, [2, 2], stride=STRIDE, padding=PADDING, \
                                     biases_initializer=None, activation_fn=tf.nn.leaky_relu, \
                                     reuse=reuse, scope='d_conv1', weights_initializer=self.initializer)


        # Conv Layer 2, batch normalization, leaky relu activation
        d2_conv = slim.convolution2d(d1_conv, 32, [2, 2], stride=STRIDE, padding=PADDING, \
                                     normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu, biases_initializer=None,\
                                     reuse=reuse, scope='d_conv2', weights_initializer=self.initializer)



        # Conv Layer 3, batch normalization, leaky relu activation
        d3_conv = slim.convolution2d(d2_conv, 64, [2, 2], stride=STRIDE, padding=PADDING, \
                                     normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu, biases_initializer=None,\
                                     reuse=reuse, scope='d_conv3', weights_initializer=self.initializer)


        # Conv Layer 3, batch normalization, leaky relu activation
        d4_conv = slim.convolution2d(d3_conv, 128, [2, 2], stride=STRIDE, padding=PADDING, \
                                     activation_fn=tf.nn.leaky_relu, biases_initializer=None,reuse=reuse, scope='d_conv4',normalizer_fn=slim.batch_norm,weights_initializer=self.initializer)


        # Conv Layer 3, batch normalization, leaky relu activation
        d5_conv = slim.convolution2d(d4_conv, 256, [2, 2], stride=STRIDE, padding=PADDING, \
                                     activation_fn=tf.nn.leaky_relu, reuse=reuse, scope='d_conv5', biases_initializer=None, weights_initializer=self.initializer)


        # Dense Layer (flatten the output), sigmoid activation
        d6_dense = slim.flatten(d5_conv, scope='d_output')


        return tf.nn.sigmoid(d6_dense), d6_dense


    def generator(self, z):

        """
        :param z: random array input dimension (batch_size, z_dim)
        :return: image (Gz)
        """

        z_, self.h0_w, self.h0_b = linear(z, 64 * 4 * 4 * 8, 'g_h0_lin', with_w=True)
        z_ = slim.batch_norm(z_)
        z_= prelu(z_)

        z_resize = tf.reshape(z_, [-1, 4, 4, 64 * 8])  # add a-relu
        # up_1 = upsampling(z_resize, [self.batch_size, 8, 8], 512, 1024, 2, name='g_up2')
        up_2 = upsampling_2D(z_resize, [self.batch_size, 8, 8], 256, 512, 2, name='g_up3', last_layer=False)
        up_3 = upsampling_2D(up_2, [self.batch_size, 16, 16], 128, 256, 2, name='g_up4', last_layer=False)
        up_4 = upsampling_2D(up_3, [self.batch_size, 32, 32], 32, 128, 2, name='g_up5', last_layer=False)
        up_5 = upsampling_2D(up_4, [self.batch_size, 64, 64], 16, 32, 2, name='g_up6', last_layer=False)
        up_6 = upsampling_2D(up_5, [self.batch_size, 128, 128], 32, 16, 2, name='g_up7', last_layer=False)
        up_7 = upsampling_2D(up_6, [self.batch_size, 256, 256], 1,32 , 2, name='g_up8', last_layer=True)

        print("Completed creating generator with last layer shape of")
        self.print_shape(up_7)

        return tf.nn.tanh(up_7)

    def discriminator_2(self,image, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            df_dim = 128  # last image dimension

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = conv2d(image, df_dim, name='d_d2_h0_conv')
            h0 = tf.nn.leaky_relu(h0)
            # h0 = tf.layers.batch_normaliztion(h0)

            h1 = conv2d(h0, df_dim * 2, name='d_d2_h1_conv')
            h1 = tf.nn.leaky_relu(h1)
            h1 = tf.layers.batch_normalization(h1)

            h2 = conv2d(h1, df_dim * 4, name='d_d2_h2_conv')  # 512
            h2 = tf.nn.leaky_relu(h2)
            h2 = tf.layers.batch_normalization(h2)

            h3 = conv2d(h2, df_dim * 8, name='d_d2_h3_conv')  # 1024
            h3 = tf.nn.leaky_relu(h3)
            h3 = tf.layers.batch_normalization(h3)

            h4 = conv2d(h3, df_dim * 16, name='d_d2_h4_conv')
            h4 = tf.nn.leaky_relu(h4)
            h4 = tf.layers.batch_normalization(h4)

            #h5 = conv2d(h4, df_dim * 32, name='d_d2_h5_conv')
            #h5 = tf.nn.leaky_relu(h5)
            #h5 = tf.layers.batch_normalization(h5)

            h5 = tf.layers.flatten(h4)

            #h6 = linear(h5, 1, 'd_d2_h5_lin')
            output = tf.nn.sigmoid(h5)

            print(h1)
            print(h2)
            print(h3)
            print(h4)
            print(h5)
            print(output)

            return output, h5

    def generator_2(self, z):

            """
            :param z: random array input dimension (batch_size, z_dim)
            :return: image (Gz)
            """

            batch_size = self.batch_size

            z_ = linear(z, 64 * 4 * 4 * 8, 'g_h0_lin')

            z_resize = tf.reshape(z_, [-1, 4, 4, 64 * 8])  # shape is [batch_sie, 4,4,512]
            z_resize = tf.nn.relu(z_resize)
            z_resize=tf.layers.batch_normalization(z_resize)

            up1 = conv2d_transpose(z_resize, [batch_size, 8, 8, 256], name='g_up1')
            up1 = tf.nn.relu(up1)
            up1 = tf.layers.batch_normalization(up1)
            up2 = conv2d_transpose(up1, [batch_size, 16, 16, 128], name='g_up2')
            up2 = tf.nn.relu(up2)
            up2 = tf.layers.batch_normalization(up2)
            up3 = conv2d_transpose(up2, [batch_size, 32, 32, 64], name='g_up3')
            up3 = tf.nn.relu(up3)
            up3 = tf.layers.batch_normalization(up3)
            up4 = conv2d_transpose(up3, [batch_size, 64, 64, 32], name='g_up4')
            up4 = tf.nn.relu(up4)
            up4 = tf.layers.batch_normalization(up4)
            up6 = conv2d_transpose(up4, [batch_size, 128, 128, 1], name='g_up6')

            """
            up5 = conv2d_transpose(up4, [batch_size, 128, 128, 16], name='g_up5')
            up5 = tf.nn.relu(up5)
            up5 = tf.layers.batch_normalization(up5)
            up6 = conv2d_transpose(up5, [batch_size, 256, 256, 1], name='g_up6')
   
            """

            print("Completed creating generator with last layer shape of")
            print(up6.shape)

            #small_image=tf.nn.tanh(conv2d_transpose(up3, [batch_size, 64, 64, 1], name='g_small_image'))

            return tf.nn.tanh(up6)






    def print_shape(self, tensor):
        print(tensor.get_shape().as_list())


    def train_fastmri(self):

            files = list(pathlib.Path(self.trainingdatadir).iterdir())
            random.shuffle(files)
            for file in files:
                centre_fraction, acceleration = get_random_accelerations()
                image, masked_kspace = get_training_pair(file, centre_fraction=centre_fraction,
                                                         acceleration=acceleration)
                print(image.shape)
                print(masked_kspace.shape)


    def train(self):

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)

                counter = 0
                learningrate = 0.0001

                for epoch in range(0, self.num_epochs):

                        print("************************ epoch:" + str(epoch) + "*****************")

                        filenames = list(pathlib.Path(self.training_datadir).iterdir())

                        np.random.shuffle(filenames)
                        print("Number training data " + str(len(filenames)))

                        np.random.shuffle(filenames)
                        Average_loss_G = 0
                        Average_loss_D = 0

                        for file in filenames:

                            centre_fraction, acceleration = get_random_accelerations()

                            #training_images: fully sampled MRI images
                            #training labels: Masked k-spaced, obtained using various mask functions, here we obtain using center_fraction =[], acceleration=[]

                            training_images, training_labels = get_training_pair(file, centre_fraction, acceleration)
                            [batch_length, x, y,z] = training_images.shape

                            for idx in range(0, batch_length, self.batch_size):


                                    z_samples = np.random.uniform(0, 256, size=(self.batch_size, self.z_dim)).astype(np.float32)

                                    batch_images = training_images[idx:idx + self.batch_size, :, :]
                                    batch_labels = training_labels[idx:idx + self.batch_size, :, :]


                                    summary1, opt, loss_D = self.sess.run(
                                        [self.merged_summary, self.OptimizerD, self.d_loss],
                                        feed_dict={self.X_train: batch_images,
                                                   self.learning_rate: learningrate,
                                                   self.z: z_samples})

                                    opt, loss_G = self.sess.run([self.OptimizerG, self.g_loss],
                                                                feed_dict={self.z: z_samples,
                                                                           self.learning_rate: learningrate})


                                    # emphrical solution to the avoid gradients vansihing issues by training generator twice, different from paper
                                    summary2, opt, loss_G = self.sess.run(
                                        [self.merged_summary, self.OptimizerG, self.g_loss],
                                        feed_dict={self.z: z_samples,
                                                   self.learning_rate: learningrate,
                                                   self.X_train: batch_images})

                                    counter += 1

                                    Average_loss_D = (Average_loss_D + loss_D) / 2
                                    Average_loss_G = (Average_loss_G + loss_G) / 2

                                    if (counter % 50 == 0):
                                        self.train_writer.add_summary(summary1, counter)
                                        self.train_writer.add_summary(summary2)

                                    if (counter % 20==0):
                                        print("Epoch: ",
                                          str(epoch) + " learning rate:" + str(learningrate) + " Generator loss:" + str(loss_G) + " Discriminator loss: " + str(loss_D))


                print("Training completed .... Saving model")
                #self.save_model(self.model_name)
                print("All completed good bye")


    def generator_test(self):

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)

                counter = 0
                learningrate = 0.0001

                for epoch in range(0, self.num_epochs):

                                    print("epoch" + str(epoch))

                                    z_samples = np.random.uniform(0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)

                                    dummy_image = np.random.uniform(0, 1.0, size=(self.batch_size, self.w, self.h, self.d)).astype(
                                        np.float32)

                                    summary2, Gz, Gz_small = self.sess.run(
                                        [self.merged_summary, self.Gz, self.Gz_small],
                                        feed_dict={self.z: z_samples,
                                                   self.learning_rate: learningrate,
                                                   self.X_train: dummy_image})

                                    counter+=1

                                    if (counter % 50 == 0):
                                        self.train_writer.add_summary(summary2, counter)










if __name__ == '__main__':


    VGG_dir = './trained_model/VGG/'
    network = DCGAN(VGG_dir, 'DCGAN')
    network.train()
