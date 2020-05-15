import h5py
import pathlib
from model.utils.subsample import MaskFunc
from model.utils.Layers import *
import model.utils.transforms as T
import tensorflow as tf


class infoGAN:
    def __init__(self, vggdir, name):
        # network parameters
        self.vggdir = vggdir
        self.training_datadir='/media/jehill/Data/ML_data/fastmri/singlecoil/train/singlecoil_train/'
        #self.labeldir=labeldir
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.num_epochs = 1000
        self.display_step = 20
        self.global_step = 0
        self.w = 256  # x
        self.h = 256  # y
        self.z_dim = 100
        self.w2 = self.w / 2  #
        self.h2 = self.h / 2  #
        self.d = 1  #  depth or channels
        self.batch_size = 20;
        self.num_classes = 20  # anging number of features to 5
        self.latent1_dim=1  #centre windows
        self.latent2_dim=1  #accelerations
        # now create the network
        self.keep_prob = 0.5  # that the drop
        self.drop_out = self.keep_prob

        # Initialize Network weights
        self.initializer = tf.truncated_normal_initializer(stddev=0.2)

        # Inputs for Discriminator and latent variable
        # [Batch,size, kspace_dim1, kspace_dim2, kspace_dim3] the final c is the k-space which we sub-sample
        # input the latent variable to create obtain the loss functions
        self.X_train = tf.placeholder(tf.float32, [None, None, None, self.d], name='X_train')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.c = tf.placeholder(tf.float32, [None, 640, 368, self.d+1])
        self.latent1=tf.placeholder(tf.float32, [None, self.latent1_dim])         #they are for centre function
        self.latent2=tf.placeholder(tf.float32, [None, self.latent2_dim])         #they are for accelerations

        self.input_image = tf.image.resize_images(self.X_train, [np.int(64), np.int(64)])
        self.c_resize = tf.image.resize_images(self.c, [np.int(640), np.int(368)])

        print(self.input_image)

        self.Gz = self.generator(self.z, self.c_resize)

        # Probabilities for real images
        self.Dx, self.Dx_logits, _ = self.discriminator(self.input_image)

        # Probabilities for generator images
        print("Discriminator Shape 2:")
        self.Dz, self.Dz_logits, self.G_z_c = self.discriminator(self.Gz, reuse=True)
        # Adversarial training using cross entropy for G and D loss, plus additional losses for infoGAN
        # Discriminator loss

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_logits, labels=tf.ones_like(self.Dx)))

        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.zeros_like(self.Dz)))
        self.d_loss = self.d_loss_fake + self.d_loss_real

        # Generator loss (adversarial)
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits, labels=tf.ones_like(self.Dz)))

        # Latent loss (from network Q)    
        self.latent_1_posterior, self.latent_2_posterior=self.Q(self.G_z_c)

        self.q_loss =self.compute_Q_loss(self.latent_1_posterior, self.latent_2_posterior, self.latent1, self.latent2)


        self.total_d_loss=self.d_loss+self.q_loss
        self.total_g_loss=self.g_loss+self.q_loss

        # get the gradients for the generator and discriminator
        self.tvars = tf.trainable_variables()
        self.d_gradients = [var for var in self.tvars if 'd_' in var.name]
        self.g_gradients = [var for var in self.tvars if 'g_' in var.name]
        self.Q_gradients = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Q')

        print(self.g_gradients)

        self.OptimizerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.total_d_loss,var_list=self.d_gradients)
        self.OptimizerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.total_g_loss,var_list=self.g_gradients)
        # summary and writer for tensorboard visulization

        # tf.summary.image("Segmentation", tf.to_float(self.segmented_image))
        tf.summary.image("Generator fake output", self.Gz)
        tf.summary.image("Input image", self.input_image, max_outputs=3)

        tf.summary.histogram("Descriminator logits (Real)", self.Dx_logits)
        tf.summary.histogram("Descriminator logits (Fake)", self.Dz_logits)

        tf.summary.scalar("Discriminator loss real", self.d_loss_real)
        tf.summary.scalar("Generator loss fake", self.d_loss_fake)
        tf.summary.scalar("Total Discriminator loss", self.d_loss)
        tf.summary.scalar("Generator loss", self.g_loss)
        #tf.summary.scalar("(Regularization loss (Q)", self.q_loss)


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
                                     biases_initializer=None, activation_fn=prelu, \
                                     reuse=reuse, scope='d_conv1', weights_initializer=self.initializer)

        # Conv Layer 2, batch normalization, leaky relu activation
        d2_conv = slim.convolution2d(d1_conv, 32, [2, 2], stride=STRIDE, padding=PADDING, \
                                     normalizer_fn=slim.batch_norm, activation_fn=prelu, \
                                     reuse=reuse, scope='d_conv2', weights_initializer=self.initializer)

        # Conv Layer 3, batch normalization, leaky relu activation
        d3_conv = slim.convolution2d(d2_conv, 64, [2, 2], stride=STRIDE, padding=PADDING, \
                                     normalizer_fn=slim.batch_norm, activation_fn=prelu, \
                                     reuse=reuse, scope='d_conv3', weights_initializer=self.initializer)

        # Conv Layer 3, batch normalization, leaky relu activation
        d4_conv = slim.convolution2d(d3_conv, 128, [2, 2], stride=STRIDE, padding=PADDING, \
                                     activation_fn=prelu, reuse=reuse, scope='d_conv4',weights_initializer=self.initializer)

        # Conv Layer 3, batch normalization, leaky relu activation
        d5_conv = slim.convolution2d(d4_conv, 256, [2, 2], stride=STRIDE, padding=PADDING, \
                                     activation_fn=prelu, reuse=reuse, scope='d_conv5', weights_initializer=self.initializer)

        d6_conv = slim.convolution2d(d5_conv, self.num_classes, [1, 1], stride=STRIDE, padding=PADDING, \
                                     activation_fn=prelu, reuse=reuse, scope='d_conv6', weights_initializer=self.initializer)  # for first working version 7 we employed d4_conv

        # Dense Layer (Fully connected), sigmoid activation
        d6_dense = slim.flatten(d6_conv, scope='d_output')

        #provide the extra output  d6_conv i.e. G(z,c) for fake images to be passed with Q network so that are reusing the graph

        return tf.nn.sigmoid(d6_dense), d6_dense, d6_conv


    def generator(self, z, c):

        """
        :param z: random array input dimension (batch_size, z_dim) : noise
        :param c: array input dimension (batch_size, kspace[0], kspace[0]) #this the randomly selected accelerated K-space, which we employ as latent vector

        :return: image (G(z,c))
        """

        # need to downsmaple kspace before combining it with noise
        d = tf.layers.dense(tf.layers.flatten(c),1024, activation=None, name='g_dense1')  #may need to change activations here
        d=tf.layers.dense(tf.layers.flatten(d),256, activation=None, name='g_dense2')  #may need to change activations here

        z=tf.concat([z,d],axis=1)
        self.print_shape(z)
        z_, self.h0_w, self.h0_b = linear(z, 64 * 4 * 4 * 8, 'g_h0_lin', with_w=True)
        z_resize = tf.reshape(z_, [-1, 4, 4, 64 * 8])  # add a-relu
        z_resize = tf.nn.relu(z_resize)

        self.print_shape(z_resize)

        up_2 = upsampling(z_resize, [self.batch_size, 8, 8], 256, 512, 2, name='g_up3')
        up_3 = upsampling(up_2, [self.batch_size, 16, 16], 128, 256, 2, name='g_up4')
        up_4 = upsampling(up_3, [self.batch_size, 32, 32], 32, 128, 2, name='g_up5')
        up_5 = upsampling(up_4, [self.batch_size, 64, 64], 1, 32, 2, name='g_up6')
        up_5 = upsampling(up_4, [self.batch_size, 64, 64], 16, 32, 2, name='g_up6')
        up_6 = upsampling(up_5, [self.batch_size, 128, 128], 32, 16, 2, name='g_up7')
        up_7 = upsampling(up_6, [self.batch_size, 256, 256], 1,32 , 2, name='g_up8')


        self.print_shape(up_7)

        return tf.nn.tanh(up_7)

    def print_shape(self, tensor):
        print(tensor.get_shape().as_list())


    def Q(self, x):

        """
        This is use to approximate Q(c|x)
        Param: x samples from generator G(z,c): i.e. the output of the generator with noise and latent code (i.e. kspace)
        output: posterior Q(c|x)
        """

        with tf.variable_scope('Q'):

            Q1 = tf.layers.dense(tf.layers.flatten(x),128)
            latent1 =  tf.layers.dense(Q1,self.latent1_dim, activation=None)
            latent2 = tf.layers.dense(Q1, self.latent2_dim, activation=None)

        return latent1, latent2  #these are (c|x) i.e.


    def compute_Q_loss(self, latent_posterior_1, latent_posterior_2, latent1, latent2):


        loss1=tf.reduce_sum(tf.squared_difference(latent_posterior_1,latent1), axis=-1)
        loss2=tf.reduce_sum(tf.squared_difference(latent_posterior_2,latent2), axis=-1)

        total_Q_loss=tf.reduce_mean(loss1)+0.5*tf.reduce_mean(loss2)

        return total_Q_loss

    def Q_loss_function(self, latent_code, discreate_c, continouse_c):

        """"
        Just for completeness from paper including discrete and continous loss function
        """
        discrete_loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.latent_code[:,:self.latent1_dim], logits=discreate_c)
        continuous_loss = tf.reduce_sum(tf.square(latent_code[:, self.categorical:] - continouse_c), axis=-1)
        total_Q_loss = tf.reduce_mean(discrete_loss) + 0.5 * tf.reduce_mean(continuous_loss)

        return total_Q_loss



    def get_training_pair(self, file, centre_fraction, acceleration):

        """


        :param file: The training image
        :param centre_fraction: randomly generated centre fraction
        :param acceleration: randomly generated
        :return:
        """


        hf = h5py.File(file)

        volume_kspace = hf['kspace'][()]
        volume_image = hf['reconstruction_esc'][()]
        mask_func = MaskFunc(center_fractions=[centre_fraction], accelerations=[acceleration])  # Create the mask function object

        volume_kspace_tensor = T.to_tensor(volume_kspace)
        masked_kspace, mask = T.apply_mask(volume_kspace_tensor, mask_func)
        masked_kspace_np=masked_kspace.numpy().reshape(masked_kspace.shape)

        return np.expand_dims(volume_image,3), masked_kspace_np



    def get_random_accelerations(self):

       """
          : we apply these to fully sampled k-space to obtain q
          :return:random centre_fractions between 0.1 and 0.001 and accelerations between 1 and 15
       """
       acceleration = np.random.randint(1, high=15, size=1)
       centre_fraction = np.random.uniform(0, 1, 1)
       decimal = np.random.randint(1, high=3, size=1)
       centre_fraction = centre_fraction / (10 ** decimal)

       return float(centre_fraction), float(acceleration)



    def train(self):

        with tf.device('/gpu:0'):

            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

                self.sess.run(self.init)

                counter = 0
                learningrate = 0.0005

                for epoch in range(0, self.num_epochs):

                        print("************************ epoch:" + str(epoch) + "*****************")

                        filenames = list(pathlib.Path(self.training_datadir).iterdir())

                        np.random.shuffle(filenames)
                        print("Number training data " + str(len(filenames)))

                        np.random.shuffle(filenames)
                        Average_loss_G = 0
                        Average_loss_D = 0

                        for file in filenames:

                            centre_fraction, acceleration = self.get_random_accelerations()

                            #training_images: fully sampled MRI images
                            #training labels: Masked k-spaced, obtained using various mask functions, here we obtain using center_fraction =[], acceleration=[]


                            #this our X and C pair
                            training_images, training_labels = self.get_training_pair(file, centre_fraction, acceleration)

                            [batch_length, x, y,z] = training_images.shape

                            print(training_images.shape)
                            print(training_labels.shape)

                            for idx in range(0, batch_length, self.batch_size):


                                batch_images = training_images[idx:idx + self.batch_size, :, :, :] #this is images X
                                batch_labels = training_labels[idx:idx + self.batch_size, :, :, :] #this is k-space


                                if (batch_labels.shape[0]==self.batch_size and batch_labels.shape[1]==640 and batch_labels.shape[2]==368):

                                    z_samples = np.random.uniform(-1, 1, size=(batch_images.shape[0], self.z_dim)).astype(
                                        np.float32)

                                    latent1=np.ones((self.batch_size,self.latent1_dim))*centre_fraction
                                    latent2 = np.ones((self.batch_size, self.latent2_dim)) * centre_fraction

                                    dict={self.X_train: batch_images,
                                                   self.c: batch_labels,
                                                   self.latent1: latent1,
                                                   self.latent2: latent2,
                                                   self.learning_rate: learningrate,
                                                   self.z:z_samples
                                                   }

                                    summary1, opt, loss_D = self.sess.run(
                                        [self.merged_summary, self.OptimizerD, self.total_d_loss],
                                        feed_dict=dict)

                                    opt, loss_G = self.sess.run([self.OptimizerG, self.total_g_loss],
                                                                feed_dict=dict)


                                    # emphrical solution to the avoid gradients vansihing issues by training generator twice, different from paper
                                    summary2, opt, loss_G = self.sess.run(
                                        [self.merged_summary, self.OptimizerG, self.total_g_loss],
                                        feed_dict=dict)

                                    counter += 1

                                    Average_loss_D = (Average_loss_D + loss_D) / 2
                                    Average_loss_G = (Average_loss_G + loss_G) / 2

                                    if (counter % 20 == 0):
                                        self.train_writer.add_summary(summary1, counter)
                                        self.train_writer.add_summary(summary2)

                                    print("Epoch: ",
                                          str(epoch) + " learning rate:" + str(learningrate) + " Generator loss:" + str(
                                              loss_G) + " Discriminator loss: " + str(loss_D))


                print("Training completed .... Saving model")
                #self.save_model(self.model_name)
                print("All completed good bye")


if __name__ == '__main__':

    VGG_dir = './trained_model/VGG/'
    network = infoGAN(VGG_dir, 'infoGAN')
    network.train()

