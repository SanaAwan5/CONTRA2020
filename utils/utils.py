import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler


def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output

def mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update

