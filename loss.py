import numpy as np
import torch
# ref) https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
def Cross_Entropy_Loss(output, target ):
	
	return (-1)*torch.sum(torch.log(torch.exp(output) / torch.sum(torch.exp(output)))*target)
	
