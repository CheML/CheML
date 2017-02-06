import csv
import os 
import numpy as np
_elements=None

def load_elements():
	atoms_fname= os.path.join(os.path.dirname(__file__),'data/atoms.csv')
	with open(atoms_fname) as fh:
		lines=list(csv.reader(fh,delimiter='\t'))[1:]
	for line in lines:
		line[0]=int(line[0])
		line[-1]=list(map(int,line[-1].split(', ')))
	return lines

def get_valence(Z):
	global _elements
	if _elements is None:
		_elements=load_elements()
	V=np.array([0]+[line[-1][-1] for line in _elements])
	return V[Z]
