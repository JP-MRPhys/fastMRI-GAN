import tensorflow as tf

import h5py
import numpy as np
import pathlib
from model.fastmri_data import get_training_pair_images_vae, get_random_accelerations
#from utils.subsample import MaskFunc
#import utils.transforms as T
#from matplotlib import pyplot as plt
from model.layers.pixelCNN import PixelCNN

class PixelVAE(tf.keras.Model):
    def __init__(self):

        super(PixelVAE, self).__init__()

        #TODO: add config parser
        #self.initizler = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

        self.training_datadir='/media/jehill/DATA/ML_data/fastmri/singlecoil/train/singlecoil_train/'

        self.BATCH_SIZE = 10
        self.num_epochs = 300
        self.learning_rate = 1e-3
        self.model_name="CVAE"

        self.image_dim = 128
        self.channels = 1
        self.latent_dim = 64

        self.kernel_size = 3
        lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.3)
        self.activation = lrelu

        self.input_image_1 = tf.placeholder(tf.float32, shape=[None, 256, 256, self.channels]) #for time being resize images
        self.input_image = tf.image.resize_images(self.input_image_1, [np.int(self.image_dim), np.int(self.image_dim)])
        self.image_shape = self.input_image.shape[1:]
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.encoder = self.inference_net()
        self.decoder = self.generative_net()  # note these are keras model

        mean, logvar = tf.split(self.encoder(self.input_image), num_or_size_splits=2, axis=1)
        self.z = self.reparameterize(mean, logvar)
        self.decoded_image = self.decoder([self.z, self.input_image])

        print(self.decoded_image)
        pixcelCNN=PixelCNN(self.decoded_image, data_type="mnist", layers=3)

        self.logits=pixcelCNN.fc2  #shape is [None,color_dim=256
        self.pixel_cnn_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(self.logits, [-1, 256]), labels=tf.cast(tf.reshape(self.input_image_1, [-1]), dtype=tf.int32)))

        #the otherway to do it via multinoimial distributions see ishaan's code for that way, but this is just more concise ?

        self.dist=tf.distributions.Categorical(logits=self.logits)
        self.pixcel_cnn_sample=self.dist.sample()

        print(self.logits)
        print(self.pixcel_cnn_sample)

        #self.reconstructed = tf.sigmoid(logits)
        # calculate the KL loss
        var = tf.exp(logvar)
        kl_loss = 0.5 * tf.reduce_sum(tf.square(mean) + var - 1. - logvar)
        #alpha variable in Ishaan' code probabaly beta VAE styple


        # cal mse loss
        sse_loss = self.pixel_cnn_loss #0.5 * tf.reduce_sum(tf.square(self.input_image - logits))
        self.total_loss = tf.reduce_mean(kl_loss + sse_loss) / self.BATCH_SIZE

        self.list_gradients = tf.trainable_variables()#self.encoder.trainable_variables + self.decoder.trainable_variables
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.total_loss, var_list=self.list_gradients)

        #see the gradient list to assess if training on relevant errors
        for var in self.list_gradients:
            print(var)

        # summary and writer for tensorboard visulization
        #tf.summary.image("Reconstructed image", self.reconstructed)
        tf.summary.image("Input image", self.input_image)
        tf.summary.scalar("KL", kl_loss)
        tf.summary.scalar("SSE",sse_loss)
        tf.summary.scalar("Total loss", self.total_loss)

        self.merged_summary = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
        self.logdir = './trained_models/' + self.model_name  # if not exist create logdir
        self.model_dir = self.logdir + 'final_model'
        print("Completed creating the model")

    def inference_net(self):
        input_image = tf.keras.layers.Input(self.image_shape)  # 224,224,1
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu')(input_image)  # 112,112,32
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu')(net)  # 56,56,64
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu')(net)  # 56,56,64
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Flatten()(net)
        # No activation
        net = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(net)
        net = tf.keras.Model(inputs=input_image, outputs=net)

        return net

    def generative_net(self):
        latent_input = tf.keras.layers.Input((self.latent_dim,))
        input_image=tf.keras.layers.Input(shape=(self.image_dim,self.image_dim,self.channels))
        net = tf.keras.layers.Dense(units=8 * 8 * 128, activation=tf.nn.relu)(latent_input)
        net = tf.keras.layers.Reshape(target_shape=(8, 8, 128))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=256,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        # No activation
        net = tf.keras.layers.Conv2DTranspose(
            filters=self.channels, kernel_size=3, strides=(1, 1), padding="SAME", activation=None)(net)
        net = tf.keras.layers.Concatenate()([net, input_image])
        upsampling_net = tf.keras.Model(inputs=[latent_input, input_image], outputs=net)
        return upsampling_net

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        # return eps * tf.exp(logvar * .5) + mean
        return eps * tf.sqrt(tf.exp(logvar)) + mean

    def train(self):
        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                learning_rate=1e-3
                counter = 0

                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)

                # so can see improvement fix z_samples
                z_samples = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, self.latent_dim)).astype(np.float32)

                for epoch in range(0, self.num_epochs):

                    print("************************ epoch:" + str(epoch) + "*****************")
                    filenames = list(pathlib.Path(self.training_datadir).iterdir())
                    np.random.shuffle(filenames)
                    print("Number training data " + str(len(filenames)))
                    np.random.shuffle(filenames)
                    for file in filenames:

                        centre_fraction, acceleration = get_random_accelerations(high=5)
                        # training_images: fully sampled MRI images
                        # training labels: , obtained using various mask functions, here we obtain using center_fraction =[], acceleration=[]
                        training_images, training_labels = get_training_pair_images_vae(file, centre_fraction, acceleration)
                        [batch_length, x, y, z] = training_images.shape

                        for idx in range(0, batch_length, self.BATCH_SIZE):

                            batch_images = training_images[idx:idx + self.BATCH_SIZE, :, :]
                            batch_labels = training_labels[idx:idx + self.BATCH_SIZE, :, :]

                            feed_dict = {self.input_image_1: batch_images,
                                         self.learning_rate: learning_rate}

                            summary, reconstructed_images, opt, loss = self.sess.run( [self.merged_summary, self.reconstructed, self.Optimizer, self.total_loss],
                                feed_dict=feed_dict)

                            sampled_image = self.sess.run(self.reconstructed, feed_dict={self.z: z_samples})

                            elbo = -loss
                            counter += 1

                            if (counter % 5 == 0):
                                self.train_writer.add_summary(summary)

                        print("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) +  "ELBO: " + str(elbo))

                print("Training completed .... Saving model")
                self.save_model(self.model_name)
                print("All completed good bye")

    def sample(self):
        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)
                # so can see improvement fix z_samples

                z_samples = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim)).astype(np.float32)

                #generate PIXEL-CNN in an autoregressive way, i.e. next pixel is dependent on current pixel
                sampled_image=np.zeros([self.batch_size,self.image_dim, self.image_dim, self.channels])

                #sampled_imaged=np.zeroes(self.BATCH_SIZE,self.image_dim, self.image_dim, self.channels)

                for i in self.image_dim:
                   for j in self.image_dim:
                      for k in self.channels:
                           #one way would be we to generate the sampled imaged with same input so how to add that needs work-out
                           sampled = self.sess.run(self.pixcel_cnn_sample, feed_dict={self.z: z_samples, self.input_image: sampled_image})
                           sampled_image[:,i,j,k]=sampled[:,i,j,k]

        return sampled_image






if __name__ == '__main__':

    model=PixelVAE()
    #model.train()
