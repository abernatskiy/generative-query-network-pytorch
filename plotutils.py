import numpy as np

def blockPlot2DTensor(tensor):
	'''
	Plots a 2D pytorch tensor with unicode block elements. Ranges:
		[0.0-0.2) - whiltespace
		[0.2-0.4) - ░
		[0.4-0.6) - ▒
		[0.6-0.8) - ▓
		[0.8-1.0] - █
	If any element falls outside of these, a ValueError will be raised
	'''
	if len(tensor.shape)!=2:
		raise ValueError(f'blockPlot2DTensor got a tensor of unexpected shape: {tensor.shape}')
	vals = tensor.numpy()
	if (vals<0.).any():
		raise ValueError('blockPlot2DTensor got a tensor with some negative values')
	if (vals>1.).any():
		raise ValueError('blockPlot2DTensor got a tensor with some values greater than one')

	symbols = [' ', '░', '▒', '▓', '█']
	indexes = np.array(vals/0.2, dtype=np.int)
	for i in range(indexes.shape[0]):
		for j in range(indexes.shape[1]):
			idx = indexes[i][j] if indexes[i][j]!=5 else 4
			print(symbols[idx], end='')
		print('')
