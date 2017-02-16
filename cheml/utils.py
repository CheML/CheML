import numpy as np 
from sklearn.utils import check_random_state
from numbers import Number

def augment(positions, translation_intervals=1,
			rotation_intervals=1, return_translations=False,
			return_rotations=False, random_state=0):
	
	if isinstance(translation_intervals,Number):
		translation_intervals=(-translation_intervals,translation_intervals)
	if len(translation_intervals)==2:
		translation_intervals=[translation_intervals]*3
	
	if isinstance(rotation_intervals,Number):
		rotation_intervals=(-rotation_intervals,rotation_intervals)
	if len(rotation_intervals)==2:
		rotation_intervals=[rotation_intervals]*3
	
	rng=check_random_state(random_state)
	translation_intervals=np.array(translation_intervals)
	translations=rng.rand(positions.shape[0],positions.shape[2])
	translations*=np.diff(translation_intervals,axis=1).ravel()
	translations+=np.array(translation_intervals)[:,0]
	
	rotation_intervals=np.array(rotation_intervals).astype(np.float)
	rotation_intervals*=(2.*np.pi/360)
	rotations=rng.rand(positions.shape[0],positions.shape[2])
	rotations*=np.diff(rotation_intervals,axis=1).ravel()
	rotations+=np.array(rotation_intervals)[:,0]

	rotation_mats2 = np.zeros((positions.shape[2],rotations.shape[0])+(2,2))
	rotation_mats = np.zeros((positions.shape[2],rotations.shape[0])+(3,3))
	rotation_mats[:,:,[0,1,2],[0,1,2]]=1
	rotation_mats2[:,:,[0,1],[0,1]] =np.cos(rotations.T)[...,np.newaxis] 
	rotation_mats2[:,:,0,1] =-np.sin(rotations.T) 
	rotation_mats2[:,:,1,0] =np.sin(rotations.T) 
	# rotation_mats2[:,:,1,1] =np.cos(rotations.T) 
	rotation_mats[0,:,:2,:2]=rotation_mats2[0]
	# rotation_mats[0,:,2,2]=1
	rotation_mats[1,:,::2,::2]=rotation_mats2[1]
	rotation_mats[2,:,1:,1:]=rotation_mats2[2]
	rmats=np.einsum('aij,ajk,akl->ail',*rotation_mats)

	rpos=np.einsum('aij,abj->abi',rmats,positions)
	rpos+=translations[:,np.newaxis,:]
	output = (rpos,)
	if return_translations:
		output= output+(translations,)
	if return_rotations:
		output= output+(rmats,)
	if len(output)==1:
		output=output[0]
	return output
