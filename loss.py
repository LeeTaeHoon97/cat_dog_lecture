import numpy as np
import torch
# ref) https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
def Binary_Cross_Entropy_Loss(output, target ):
	
	return torch.mean(-1* (target*torch.log(output)+(1-target)*torch.log(1-output)))
	
