import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D,Input,Conv2D,Conv2DTranspose,LeakyReLU,Activation,Concatenate,Dropout,BatchNormalization
from keras.models import Model
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
import glob
from skimage import io
from PIL import Image
import cv2
'''
pix2pix GAN model
Based on the code by Jason Brownlee from his blogs on https://machinelearningmastery.com/
Original paper: https://arxiv.org/pdf/1611.07004.pdf

encoder:
C64-C128-C256-C512-C512-C512-C512-C512
decoder:
CD512-CD512-CD512-C512-C256-C128-C64

Discriminator:
C64-C128-C256-C512
'''

def Discriminator(image_shape=(256,256,3)):
    
	# weight initialization
	init = RandomNormal(stddev=0.02) 

	input_img = Input(shape=image_shape)  
	input_target = Input(shape=image_shape)  #Image generated after training. 

	merged = Concatenate()([input_img, input_target])
    
	# C64
	x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	x = LeakyReLU(alpha=0.2)(x)
	# C128
	x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	# C256
	x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	# C512
	x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	# Stride 1x1
	x = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	# output
	x = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)
	x = Activation('sigmoid')(x)
	# define model
	model = Model([input_img, input_target], x)
    
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5) #learning rate and 0.5 beta
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

def Encoder(input_layer, filters):

	init = RandomNormal(stddev=0.02)
	x = Conv2D(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_layer)
	x = BatchNormalization()(x, training=True)
	x = LeakyReLU(alpha=0.2)(x)
	return x


def Decoder(input_layer, skip_conn, filters, dropout=True):

	init = RandomNormal(stddev=0.02)
	x = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_layer)
	x = BatchNormalization()(x, training=True)
	if dropout:
		x = Dropout(0.5)(x, training=True)
	x = Concatenate()([x, skip_conn])
	x = Activation('relu')(x)
	return x

def Generator(image_shape=(256,256,3)):

	init = RandomNormal(stddev=0.02)
	input_img = Input(shape=image_shape)
    
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_img)
	e1 = LeakyReLU(alpha=0.2)(e1)
	e2 = Encoder(e1, 128)
	e3 = Encoder(e2, 256)
	e4 = Encoder(e3, 512)
	e5 = Encoder(e4, 512)
	e6 = Encoder(e5, 512)
	e7 = Encoder(e6, 512)
	# bottleneck
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD512-CD512-C512-C256-C128-C64
	d1 = Decoder(b, e7, 512)
	d2 = Decoder(d1, e6, 512)
	d3 = Decoder(d2, e5, 512)
	d4 = Decoder(d3, e4, 512, dropout=False)
	d5 = Decoder(d4, e3, 256, dropout=False)
	d6 = Decoder(d5, e2, 128, dropout=False)
	d7 = Decoder(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7) #Modified 
	g = Activation('tanh')(g)

	model = Model(input_img, g)
	return model

def GANs(g_model, d_model, image_shape):
	# discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False       

	source = Input(shape=image_shape)
	gen_out = g_model(source)
	# supply the input image and generated image as inputs to the discriminator
	dis_out = d_model([source, gen_out])

	model = Model(source, [dis_out, gen_out])
	opt = Adam(lr=0.0002, beta_1=0.5)
    #Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE)
    #Authors suggested weighting BCE vs L1 as 1:100.
	model.compile(loss=['binary_crossentropy', 'mae'],optimizer=opt, loss_weights=[1,100])
    
	return model


def observe_performance(step, g_model,image,mask):

    ix = np.random.randint(0, image.shape[0], 1)
    I=image[ix]
    M=mask[ix]
    G_fake_M=g_model.predict(M)
	# scale all pixels from [-1,1] to [0,1]
    i = (I + 1) / 2.0
    m = (M + 1) / 2.0
    gfm = (G_fake_M + 1) / 2.0
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(m[0,:,:,:])
    plt.subplot(1,3,2)
    plt.imshow(gfm[0,:,:,:])
    plt.subplot(1,3,3)
    plt.imshow(i[0,:,:,:])
    figure_name = './result/plot_%06d.png' % (step+1)
    plt.savefig(figure_name)
    plt.close()
    
    filename = './model_%06d.h5' % (step+1)
    g_model.save(filename)
    print('>Saved: %s and %s' % (figure_name, filename))

def train(d_model, g_model, gan_model,image,mask, epochs=60):
    n_steps = len(image) * epochs
    n_patch = d_model.output_shape[1]
    for i in range(n_steps):
        ix = np.random.randint(0, image.shape[0], 1)
        x_real=image[ix]
        mask_real=mask[ix]
        y_real = np.ones((1, n_patch, n_patch, 1))
        
        x_fake=g_model.predict(mask_real)
        y_fake=np.zeros((1, n_patch, n_patch, 1))
        d_loss1 = d_model.train_on_batch([mask_real,x_real], y_real)
		# update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([mask_real, x_fake], y_fake)
		# update the generator
        g_loss, _, _ = gan_model.train_on_batch(mask_real, [y_real, x_real])
		# summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
        if (i+1) % (len(image) * 10) == 0:
            observe_performance(i, g_model,image,mask)
          
###############################################################################
img_path=glob.glob('./Train/images/*')
mask_path=glob.glob('./Train/masks/*')

I,M=[],[]
for i in range(len(img_path)):
    img=io.imread(img_path[i])
    img=(img-127.5)/127.5
    mask=io.imread(mask_path[i])
    mask=(mask-127.5)/127.5
    I.append(img)
    M.append(mask)
    
I=np.array(I)
M=np.array(M)

D = Discriminator()
G = Generator()
gan_model = GANs(G, D, image_shape=(256,256,3))

from datetime import datetime 
start1 = datetime.now() 
train(D,G, gan_model,I,M) 
stop1 = datetime.now()
execution_time = stop1-start1
print("Execution time is: ", execution_time)






