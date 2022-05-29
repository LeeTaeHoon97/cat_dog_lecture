import numpy as np
# ref) https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
def Cross_Entropy_Loss(output, target ):
	return	np.sum(np.log(np.exp(output) / np.sum(np.exp(output)))*target)

