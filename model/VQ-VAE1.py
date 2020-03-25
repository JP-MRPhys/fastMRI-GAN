import tensorflow as tf

import h5py
import numpy as np
import pathlib
from utils.subsample import MaskFunc
import utils.transforms as T
#from matplotlib import pyplot as plt
from model.fastmri_data import get_training_pair_images_vae, get_random_accelerations
from model.layers.vector_quantizier import vector_quantizer
from model.layers.PixelCNN2 import pixelcnn

#test

grad_clip_pixelcnn=1
learning_rate_pixelcnn= 1e-3
learning_rate=3e-4



class VQ_VAE1(tf.keras.Model):
    def __init__(self):

        super(VQ_VAE1, self).__init__()

        #TODO: add config parser
        #self.initizler = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
        self.training_datadir='/media/jehill/DATA/ML_data/fastmri/singlecoil/train/singlecoil_train/'

        self.BATCH_SIZE = 10
        self.num_epochs = 300
        self.learning_rate = 1e-3
        self.model_name="CVAE"

        self.image_dim = 128
        self.channels = 1
        self.latent_dim = 64    #embedding dimensions D
        self.num_embeddings=8   # k: Categorical, increasing this is recommeded compared to the bottle neck layer
        self.code_size= 16 #to be set directly from z_e

        self.commitment_cost=0.25

        self.kernel_size = 3
        lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.3)
        self.activation = lrelu


        #Place holders
        self.input_image_1 = tf.placeholder(tf.float32, shape=[None, 256, 256, self.channels]) #for time being resize images
        self.input_image = tf.image.resize_images(self.input_image_1, [np.int(self.image_dim), np.int(self.image_dim)])
        self.image_shape = self.input_image.shape[1:]

        self.encoder = self.inference_net()
        self.decoder = self.generative_net()  # note these are keras model

        self.z_e = self.encoder(self.input_image)
        self.code_size=(self.z_e.shape)[1]
        self.z_samples=tf.placeholder(tf.float32, shape=self.z_e.shape) #placeholder for recons/samples
        vq=vector_quantizer(self.z_e,self.latent_dim, self.num_embeddings, self.commitment_cost)
        z_q=vq['quantized']
        self.pixelCNN_train_input=vq['encoding_indices']  # placeholder for separate training of pixelCNN prior i.e. encoding indices #shape is (batch, code_size, code_size)

        logits = self.decoder(z_q)
        logits = tf.sigmoid(logits)
        self.reconstruction=tf.nn.sigmoid(logits)

        # cal mse loss
        sse_loss =  tf.reduce_sum(tf.square(self.input_image - self.reconstruction))
        self.total_loss = sse_loss + vq['loss']
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.total_loss)

        #pixelCNN
        #pixelCNN inputs
        self.pixelCNN_input=tf.placeholder(tf.float32, shape=(None, self.code_size, self.code_size))
        self.pixelCNN_samples=tf.placeholder(tf.float32, shape=(None, self.code_size, self.code_size))
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        #pixel CNN
        self.pixelcnn=pixelcnn(self.pixelCNN_input, num_layers_pixelcnn=12, fmaps_pixelcnn=32, num_embeddings=self.num_embeddings, code_size=self.code_size)
        self.loss_pixelcnn = self.pixelcnn["loss_pixelcnn"]
        self.sampled_pixelcnn_train = self.pixelcnn["sampled_pixelcnn"]


        self.trainer_pixelcnn = tf.train.RMSPropOptimizer(learning_rate=learning_rate_pixelcnn)
        gradients_pixelcnn = self.trainer_pixelcnn.compute_gradients(self.loss_pixelcnn)
        clipped_gradients_pixelcnn = map(
            lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -grad_clip_pixelcnn, grad_clip_pixelcnn),
                                                 gv[1]], gradients_pixelcnn)
        # clipped_gradients_pixelcnn = [(tf.clip_by_value(_[0], -grad_clip_pixelcnn, grad_clip_pixelcnn), _[1]) for _ in gradients_pixelcnn]
        self.optimizer_pixelcnn = self.trainer_pixelcnn.apply_gradients(clipped_gradients_pixelcnn)

        #reconstructions
        vq_recons=vector_quantizer(self.z_samples, embedding_dim=self.latent_dim,num_embeddings=self.num_embeddings, commitment_cost=self.commitment_cost, only_lookup=True,inputs_indices=self.pixelCNN_samples)
        self.recon_pixelcnn=tf.sigmoid(self.decoder(vq_recons['quantized']))

        # TODO: add summaries
        # summary and writer for tensorboard visulization
        #tf.summary.image("Reconstructed image", self.recon_pixelCNN)
        #tf.summary.image("Input image", self.input_image)
        #tf.summary.scalar("SSE",sse_loss)
        #tf.summary.scalar("Total loss", self.total_loss)

        self.merged_summary = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()

        self.logdir = './' + self.model_name  # if not exist create logdir
        self.model_dir = self.logdir + 'final_model'

        print("Completed creating the model")

    def inference_net(self):
        input_image = tf.keras.layers.Input(self.image_shape)  # 128,128,1
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(input_image)  # 64,64,32
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=4, strides=(2, 2), activation='relu', padding='same')(net)  # 32,32,64
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same')(net)  # 16,16,64
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=self.latent_dim, kernel_size=1, strides=(1, 1), activation='relu')(net) # w/4,h/4,latent_dim
        net = tf.keras.Model(inputs=input_image, outputs=net)

        #residual connections not implemented as per the paper
        return net

    def generative_net(self):
        latent_input = tf.keras.layers.Input((self.image_dim/4,self.image_dim/4,self.latent_dim))
        net = tf.keras.layers.Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation=self.activation)(latent_input)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        # No activation
        net = tf.keras.layers.Conv2DTranspose(
            filters=self.channels, kernel_size=1, strides=(1, 1), padding="SAME", activation=None)(net)
        upsampling_net = tf.keras.Model(inputs=latent_input, outputs=net)
        return upsampling_net

    # we train both pixelCNN and the VQ-VAE
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

                            feed_dict = { self.input_image_1: batch_images,
                                          self.learning_rate: learning_rate }

                            #train VQ-VAE
                            summary, reconstructed_images, opt, loss, pixelcnn_training_input = self.sess.run([self.merged_summary, self.reconstruction, self.Optimizer, self.total_loss, self.pixelCNN_train_input],
                                feed_dict=feed_dict)

                            #train PixelCNN
                            feed_dict={self.input_image_1: self.batch_input,  self.pixelCNN_input: pixelcnn_training_input}
                            pixelcnn_loss, _= self.sess.run(self.loss_pixelcnn, self.optimizer_pixelcnn, feed_dict)

                            #sampled_image = self.sess.run(self.reconstructed, feed_dict={self.z: z_samples})
                            elbo = -loss

                            counter += 1
                            if (counter % 5 == 0):
                                self.train_writer.add_summary(summary)

                        print("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) +  "ELBO: " + str(elbo) + "VQ-VAE loss" + str(loss))
                        print("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) +  "Pixel CNN Loss: " + str(pixelcnn_loss))

                print("Training completed .... Saving model")
                # self.save_model(self.model_name)
                print("All completed good bye")


    def train_pixelCNN(self):
            #Not tested just if you want a separate loop but will be 2X longer
            with tf.device('/gpu:0'):
              with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                learning_rate=1e-3
                counter = 0


                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)

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

                            pixelcnn_training_input= self.sess.run(self.pixelCNN_train_input, feed_dict)
                            feed_dict={self.input_image_1: batch_images,  self.pixelCNN_input: pixelcnn_training_input}
                            pixelcnn_loss, _= self.sess.run(self.loss_pixelcnn, self.optimizer_pixelcnn, feed_dict)

                            counter += 1

                        print("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) +  "Pixel CNN Loss: " + str(pixelcnn_loss))


                print("Training completed .... Saving model")
                # self.save_model(self.model_name)
                print("All completed good bye")    


    def reconstruct_withPixelCNN(self, test_image):

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)
                #generate the prior's first
                samples = self.generate_PixelCNN_samples(self.sess, self.image_dim, self.image_dim)

                #pass x encoder to z and then z and with pixelCNN sample to recon image
                feed_dict = {self.input_image: test_image}
                z=self.sess.run(self.z_e, feed_dict=feed_dict)

                #pass it to reconstructure the recon imageS
                feed_dict = {self.z_samples: z, self.pixelCNN_samples: samples}      # feed_dict = {self.z_samples: z, self.sampled_code_pixelcnn: samples}
                recon_pixelcnn_res = self.sess.run(self.recon_pixelcnn, feed_dict=feed_dict)

                return recon_pixelcnn_res

    def reconstruct(self, test_image):

        number_images=np.shape(test_image)[0]  #asumming batch_size, H,W,C
        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)
                # generate the prior's i.e. encoding indices from a gaussian
                prior_samples = np.random.randint(1, self.num_embeddings, size=(number_images, self.code_size, self.code_size)).astype(np.float32)   # TODO: recheck the flow here

                # pass x encoder to z and then z and with pixelCNN sample to recon image
                feed_dict = {self.input_image: test_image}
                z = self.sess.run(self.z_e, feed_dict=feed_dict)

                # pass it to reconstructure the recon imageS
                feed_dict = {self.z_samples: z,
                             self.pixelCNN_samples: prior_samples}  # feed_dict = {self.z_samples: z, self.sampled_code_pixelcnn: samples}
                recon_pixelcnn_res = self.sess.run(self.recon_pixelcnn, feed_dict=feed_dict)

                return recon_pixelcnn_res

    def images_samples_withPixelCNN(self, number_of_images):


        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)

                #generate the prior's first these are sof the shape (number_images, seld.code_size, self.code_size)
                prior_indices_pixelCNN = self.generate_PixelCNN_samples(self.sess, number_of_images)

                #- this should be z_e shape
                z_e= np.random.uniform(-1, 1, size=(number_of_images, self.code_size, self.code_size, self.latent_dim)).astype(np.float32)  #TODO: may need to check the shapes current assuming 1 images
                #First we sample prior encoding_indices from pixelCNN shape (number_of_sample=, code_size, code_size)
                # we then pass fake "z_e" to obtain "z_q" from these encoding indices via look i.e z_e not used, then z_q which is feed to cnn-decoder

                feed_dict = {self.z_samples: z_e, self.pixelCNN_samples: prior_indices_pixelCNN}
                recon_pixelcnn_res = self.sess.run(self.recon_pixelcnn, feed_dict=feed_dict)

                # #Option 2 can also use text_image=np.zeros(shape=(batch, self.image_dim,self.image_dim,self.channels) to get z_e
                # Option 3 generated random z_q, see VQ code & reference for info

        return recon_pixelcnn_res

    def images_samples(self, number_of_images):

        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)
                # generate the prior's first incideces first
                prior_indices=np.random.randint(0, self.num_embeddings, shape=(number_of_images, self.code_size, self.code_size))

                # - this should be z_e shape
                z_e = np.random.uniform(-1, 1, size=(
                number_of_images, self.code_size, self.code_size, self.latent_dim)).astype(
                    np.float32)
                # First we sample prior encoding_indices from pixelCNN shape (number_of_sample=, code_size, code_size)
                # we then pass fake "z_e" to obtain "z_q" from prior encoding indices via look i.e z_e not used, then z_q which is feed to cnn-decoder

                feed_dict = {self.z_samples: z_e, self.pixelCNN_samples: prior_indices}
                recon_pixelcnn_res = self.sess.run(self.recon_pixelcnn, feed_dict=feed_dict)

                # Option 2 can also use text_image=np.zeros(shape=(batch, self.image_dim,self.image_dim,self.channels) to get z_e
                # Option 3 generated random z_q, see VQ code & reference for info

        return recon_pixelcnn_res


    def generate_PixelCNN_samples(self, sess, batch_size):

        samples = np.zeros(shape=(batch_size, self.code_size, self.code_size), dtype=np.int32)
        for j in range(self.code_size):
            for k in range(self.code_size):
                data_dict = {self.pixelCNN_samples: samples}
                next_sample = sess.run(self.sampled_pixelcnn_train, feed_dict=data_dict)
                samples[:, j, k] = next_sample[:, j, k]
        samples.astype(np.int32)

        return samples
        





if __name__ == '__main__':

    model=VQ_VAE1()
    #model.train()
