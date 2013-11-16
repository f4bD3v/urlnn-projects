import Image
import glob, os
import numpy as np
from sklearn.feature_extraction import image

eta = 5E-6
tao = 1200
theta = 5

def posrect(pij):
	if pij > 0:
		return pij
	else:
		return 0

def negrect(pij):
	if pij < 0:
		return abs(pij)
	else:
		return 0

def init_weights(mu_w0, sigma_w0, num_weights):
	w = [np.random.normal(mu_w0, sigma_w0) for elem in range(0,256)]
	return np.array(w).reshape(16,16)

def time_step(X_ON_set, X_OFF_set):
	X_ON = np.random.choice(X_ON_set)
	X_OFF = np.random.choice(X_OFF_set)

	# output neuron activity
	dimij = len(X_ON)
	y = pr(np.dot(np.reshape(W_ON, dimij), np.reshape(X_ON, dimij).T)+np.dot(np.reshape(W_OFF, dimij), np.reshape(X_OFF, dimij).T))

	'''
	dtheta=(pow(y,2)-theta)/tao
	ntheta=theta+dtheta
	d_W_ON =eta*np.dot(X_ON,((pow(y,2)-np.dot(y, ntheta))
	wn=w+dw
	'''

	pr(W_ON)
	pr(W_OFF)

	return

def main():
	pr = np.vectorize(posrect)
	nr = np.vectorize(negrect)

	imgs = glob.glob("img/*.bmp")
	img_arrs = [np.array(Image.open(img)) for img in imgs]
	print img_arrs
	mus = [np.mean(img) for img in img_arrs]
	sigmas = [pow(np.std(img),2) for img in img_arrs]
	zmu_norms = [(img_arrs[i]-mus[i])/sigmas[i] for i in range(0,len(img_arrs))]

	# using scikit package for easy patch extraction
	patches_list = [image.extract_patches_2d(zmu_norm, (16,16), max_patches = 5000, random_state = 0) for zmu_norm in zmu_norms]
	patches = np.vstack(patches_list)

	# for each patch compute the activities of the presynaptic cells xij_ON, xij_OFF
	# from 50000 patches construct 10000 input vectors
	X_ON_set = pr(patches) 
	X_OFF_set = nr(patches) 
	print X_ON_set

	mu_w0 = 0.5
	sigma_w0 = 0.15
	w_ON = init_weights(mu_w0, sigma_w0, 256)
	w_OFF = init_weights(mu_w0, sigma_w0, 256)



if __name__ == "__main__":
	main()