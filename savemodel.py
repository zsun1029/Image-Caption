import os
import torch


 
model = torch.load('info_3_batch_16train_2.59e+01_e_49_en.pt')
 
torch.save(model.state_dict(), 'info_3_batch_16train_2.59e+01_e_49_state_dict.pt') 
