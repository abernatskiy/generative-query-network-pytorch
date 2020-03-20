import os
import GPUtil

currentGPUs = []

def selectGPU(numGPUs, maxMemUsage=0.01, maxProcUsage=0.01, messageOnSuccess=True):
	global currentGPUs
	for gpu in GPUtil.getGPUs():
		if gpu.memoryUtil<maxMemUsage and gpu.load<maxProcUsage:
			currentGPUs.append(gpu.id)
		if len(currentGPUs)>=numGPUs:
			os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, currentGPUs))
			if messageOnSuccess:
				print(f'Selected GPU(s): {os.environ["CUDA_VISIBLE_DEVICES"]}')
			return
	raise ConnectionError(f'No available GPUs satisfy maxMemUsage={maxMemUsage} and maxProcUsage={maxProcUsage}')

def getMemUsage(absolute=False):
	allgpus = GPUtil.getGPUs()
	selectedgpus = [ allgpus[id] for id in currentGPUs ]
	if absolute:
		return [ gpu.memoryUsed for gpu in selectedgpus ]
	else:
		return [ gpu.memoryUtil for gpu in selectedgpus ]

def printMemUsage(absolute=True, prefix='Current memory usage: '):
	if absolute:
		ustrs = [ f'gpu{gpuid}:{memu:.0f}MiB' for gpuid, memu in zip(currentGPUs, getMemUsage(absolute=True)) ]
	else:
		ustrs = [ f'gpu{gpuid}:{memu:.4f}' for gpuid, memu in zip(currentGPUs, getMemUsage(absolute=False)) ]
	print(prefix + ' '.join(ustrs))
