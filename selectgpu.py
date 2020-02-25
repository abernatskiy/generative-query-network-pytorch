import os
import GPUtil

currentGPUs = []

def selectGPU(numGPUs, maxMemUsage=0.01, maxProcUsage=0.01, messageOnSuccess=True):
	global currentGPUs
	for gpu in GPUtil.getGPUs():
		if len(currentGPUs)>=numGPUs:
			os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, currentGPUs))
			if messageOnSuccess:
				print(f'Selected GPU(s): {os.environ["CUDA_VISIBLE_DEVICES"]}')
			return
		if gpu.memoryUtil<maxMemUsage and gpu.load<maxProcUsage:
			currentGPUs.append(gpu.id)
	raise ConnectionError(f'No available GPUs satisfy maxMemUsage={maxMemUsage} and maxProcUsage={maxProcUsage}')

def getMemUsage():
	allgpus = GPUtil.getGPUs()
	selectedgpus = [ allgpus[id] for id in currentGPUs ]
	return [ gpu.memoryUtil for gpu in selectedgpus ]
