import numpy as np 
from sklearn.utils import check_random_state
from numbers import Number
import csv
import os 
_elements=None

def _load_elements():
	"""
	Loads chemical element information.
	
	Returns:
	    list: list of lists with the following entries [nuclear charge, element name, valence] for each element
	"""

	atoms_fname= os.path.join(os.path.dirname(__file__),'data/atoms.csv')
	import ast

	with open(atoms_fname) as fh:
		lines=list(csv.reader(fh,delimiter='\t'))[1:]
	for line in lines:
		line[0]=int(line[0]) #Nuclear Charge
		# line[-1]=list(map(int,line[-1].split(', ')))
		line[2]=list(ast.literal_eval(line[2])) #Electorns per shell
		line[3]=int(line[3]) #Empirical Radius of the atom
		line[4]=int(line[4]) #Calculated Radius of the atom
		line[5]=int(line[5]) #van der Waals based radius
		line[6]=int(line[6]) #Covalent radius (single bond)
		line[7]=int(line[7]) #Covalent radius (triple bond)
	return lines

def get_valence(Z):
	"""
	Converts array of nuclear charges to array of corresponding valence.

	Args:
	    Z (numpy ndarray): array with nuclear charges
	
	Returns:
	    numpy ndarray: array of the same size as Z with the valence of the corresponding atom 
	"""
	global _elements
	if _elements is None:
		_elements=_load_elements()
	V=np.array([0]+[line[2][-1] for line in _elements])
	return V[Z]

def get_calculated_radius(Z):
	"""
	Converts array of nuclear charges to array of corresponding valence.

	Args:
	    Z (numpy ndarray): array with nuclear charges
	
	Returns:
	    numpy ndarray: array of the same size as Z with the valence of the corresponding atom 
	"""
	global _elements
	if _elements is None:
		_elements=_load_elements()
	V=np.array([0]+[line[4] for line in _elements])
	ret=V[Z]/100. #angstrom conversion
	# assert (ret>0).all()
	return ret

def get_van_der_waals_radius(Z):
	"""
	Converts array of nuclear charges to array of corresponding valence.

	Args:
	    Z (numpy ndarray): array with nuclear charges
	
	Returns:
	    numpy ndarray: array of the same size as Z with the valence of the corresponding atom 
	"""
	global _elements
	if _elements is None:
		_elements=_load_elements()
	V=np.array([0]+[line[5] for line in _elements])
	ret=V[Z]/100. #angstrom conversion
	# assert (ret>0).all()
	return ret



def get_empirical_radius(Z):
	"""
	Converts array of nuclear charges to array of corresponding valence.

	Args:
	    Z (numpy ndarray): array with nuclear charges
	
	Returns:
	    numpy ndarray: array of the same size as Z with the valence of the corresponding atom 
	"""
	global _elements
	if _elements is None:
		_elements=_load_elements()
	V=np.array([0]+[line[3] for line in _elements])
	ret=V[Z]/100. #angstrom conversion
	# assert (ret>0).all()
	return ret


def augment(positions, translation_intervals=1,
			rotation_intervals=1, return_translations=False,
			return_rotations=False, random_state=0):
	"""Creates new molecules by applying random rotations and translations to the given atom positions.
	
	Args:
	    positions (numpy ndarray NxMx3): array of M atom positions for N molecules
	    translation_intervals (int|list, optional): the translations will be made by a uniformly drawn sample in the interval defined by the list or [-number,number] 
	    rotation_intervals (number|list, optional): the rotations will be made by a uniformly drawn sample (degrees) in the interval defined by the list or [-number,number] 
	    return_translations (bool, optional): if True it returns the translation offsets for each molecule
	    return_rotations (bool, optional): if True it returns the rotation matrices used to rotate each molecule
	    random_state (int, optional): Initial state for random number generator
	
	Returns:
	    numpy ndarray: array of the same shape as positions with new positions
	    numpy ndarray, optional: [N,3] with the applied translations
	    numpy ndarray, optional: [N,3,3] with the applied translations
	"""
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
