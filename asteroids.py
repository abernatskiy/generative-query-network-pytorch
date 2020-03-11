import gzip
import numpy as np
from pathlib import Path

import torch

#def transform_viewpoint(v):
#	'''Transforms the viewpoint vector into a consistent
#	representation
#	'''
#	w, z = torch.split(v, 3, dim=-1)
#	y, p = torch.split(z, 1, dim=-1)
#
#	# position, [yaw, pitch]
#	view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
#	v_hat = torch.cat(view_vector, dim=-1)
#
#	return v_hat

class Asteroids(torch.utils.data.Dataset):
	'''
	Dataset of rendered snapshots of asteroid shapes.

	:param rootDir: location of data on disc
	:param train: whether to use train of test set
	:param numChunks: how many chunks to use (training chunks are taken from the beginning of the set, testing - from its end)
	:param totalChunks: how many chunks in total are there in the dataset
	:param chunkSize: size of a chunk of shapes
	'''
	def __init__(self, rootDir, train, numChunks, totalChunks=200, chunkSize=320):
		super(Asteroids, self).__init__()
		self.rootDir = rootDir
		assert numChunks <= totalChunks
		self.startAtChunk = 0 if train else totalChunks-numChunks
		self.numChunks = numChunks
		self.chunkSize = chunkSize

	def getChunk(self, idx):
		majorChunk = self.startAtChunk + (idx // self.chunkSize)
		if majorChunk >= self.startAtChunk + self.numChunks:
			raise IndexError(f'Index {idx} is out of range as it corresponds to a major chunk {majorChunk} (only chunks {self.startAtChunk}-{self.startAtChunk+self.numChunks} are available for this dataset)')
		minorChunk = idx % self.chunkSize
		return majorChunk, minorChunk

	def __len__(self):
		return self.numChunks*self.chunkSize

	def __getitem__(self, idx):
		majorChunk, minorChunk = self.getChunk(idx)

		minorChunkFilePath = self.rootDir / f'chunk{majorChunk}' / f'asteroidChunk{minorChunk}.pt.gz'
		with gzip.open(minorChunkFilePath, 'r') as f:
			images, viewpoints = torch.load(f)

		# uint8 -> float32
		images = torch.FloatTensor(images)/255
		viewpoints = torch.FloatTensor(viewpoints)

		return images, viewpoints

