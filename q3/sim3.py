import Image
import glob, os
import numpy as np
from sklearn.feature_extraction import image

import matplotlib.pyplot as plt

eta = 5E-6
tao = 1200

def posrect(pij):
	if pij > 0:
		return pij
	else:
		return 0.0

def negrect(pij):
	if pij < 0:
		return np.fabs(pij)
	else:
		return 0.0

pr = np.vectorize(posrect)
nr = np.vectorize(negrect)

def init_weights(mu_w0, sigma_w0, num_weights):
	w = [np.random.normal(mu_w0, sigma_w0) for elem in range(0,num_weights)]
	return np.array(w)

def time_step(w, x_ON, x_OFF, theta):
	w_ON = w[0]
	w_OFF = w[1]

	# X's and W's are given as row vectors
	# output neuron activity
	y = pr(np.dot(w_ON, x_ON)+np.dot(w_OFF, x_OFF))

	# update theta
	dtheta=(pow(y,2)-theta)/tao
	ntheta=theta+dtheta

	# update w_ON
	dw_ON = np.multiply(eta*(pow(y,2)-y*ntheta), x_ON)
	w_ON_n = w_ON+dw_ON
	# constrain
	w_ON = pr(w_ON_n)

	# update W_OFF
	dw_OFF = np.multiply(eta*(pow(y,2)-y*ntheta), x_OFF)
	w_OFF_n = w_OFF+dw_OFF
	# constrain
	w_OFF = pr(w_OFF_n)

	W = np.reshape(w_ON, (16,16))-np.reshape(w_OFF, (16,16))
	theta = ntheta

	return (W, np.vstack((w_ON, w_OFF)), theta)

def main():

	imgs = glob.glob("img/*.bmp")
	img_arrs = [np.array(Image.open(img)) for img in imgs]

	mus = [np.mean(img) for img in img_arrs]
	sigmas = [np.std(img) for img in img_arrs]
	zmu_norms = [(img_arrs[i]-mus[i])/sigmas[i] for i in range(0,len(img_arrs))]
	# would have been cleaner using scikit

	# using scikit package for easy patch extraction
	patches_list = [[np.reshape(patch, 256) for patch in image.extract_patches_2d(zmu_norm, (16,16), max_patches = 5000, random_state = 0)] for zmu_norm in zmu_norms]
	patches = np.vstack(patches_list)

	# for each patch compute the activities of the presynaptic cells xij_ON, xij_OFF
	# from 50000 patches construct 100000 input vectors
	x_ON_set = pr(patches) 
	x_OFF_set = nr(patches) 

	theta = 5

	mu_w0 = 0.5
	sigma_w0 = 0.15
	w_ON = init_weights(mu_w0, sigma_w0, 256)
	w_OFF = init_weights(mu_w0, sigma_w0, 256)

	w = np.vstack((w_ON, w_OFF))

	y, x = np.mgrid[slice(0, 16 + 1, 1),
                	slice(0, 16 + 1, 1)]


	os.chdir("output")

	titer = 150000
	for niter in xrange(1, titer+1):
		i = np.random.choice(range(0,50000))
		x_ON = x_ON_set[i]
		x_OFF = x_OFF_set[i]

		ret = time_step(w, x_ON, x_OFF, theta)
		W = ret[0]
		w = ret[1]
		theta = ret[2]

		if niter%5000==0 or niter==1:
			print "iter to go: "+str(niter)
			plt.pcolor(x, y, W, cmap='RdBu', vmin=np.amin(W), vmax=np.amax(W))
			plt.title("Receptive Field")
			# set limits of axis to limits of data?
			#plt.axis([x.min()])
			plt.colorbar()
			plt.savefig("receptive_field_"+str(niter), bbox_inches=0)
			plt.clf()

	return


if __name__ == "__main__":
	main()
