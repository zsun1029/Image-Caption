from __future__ import print_function
import argparse
import time
import os
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.nn.functional as F
from torch.autograd import Variable # AttributeError: Can't get attribute 'Vocabulary' on <module '__main__' from 'code-CnnRnn/xwgeng/main.py'>
import torch.optim as optim # from torch.optim.lr_scheduler import LambdaLR
from modelgru import Encoder2Decoder
import pickle
from util import get_loader, get_metrics # mt_iterator, get_bleu, filter_bleu   
from build_vocab import Vocabulary
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import json
import math

parser = argparse.ArgumentParser(description='Sequence-to-Tree model')
# data 
parser.add_argument('--max_size', type=int, default=30000, help='minimum size of word in vocab')
parser.add_argument('--curriculum', type=int, default=2, help='curriculum learning') 
# model
parser.add_argument('--name', type=str, default='', help='the name of checkpoint')
parser.add_argument('--ninp', type=int, default=512, help='size of word embedding')
parser.add_argument('--nhid', type=int, default=512, help='number of hidden layer')
# optimization
parser.add_argument('--optim', type=str, default='Adam', help='optimization algorihtim')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate') 
parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
# bookkeeping
parser.add_argument('--seed', type=int, default=123, help='random number seed')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to save the model')
# GPU
parser.add_argument('--cuda', action='store_true', help='use cuda')
# Misc
parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train')
parser.add_argument('--epoch', type=int, default=42, help='epoch of checkpoint')
parser.add_argument('--info', type=str, help='info of the model')
# test
parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size for test')
parser.add_argument('--beam_size', type=int, default=10, help='size of beam')
parser.add_argument('--pred_max_len', type=int, default=200, help='maximum length of generated sequence')
parser.add_argument('--save', type=str, default='./generation/', help='path to save generated sequence')
parser.add_argument('--cache', type=str, default='./cache/', help='path to save log file')
# 
parser.add_argument('--imageDir', type=str, default='/users4/zsun/pytorch/image_caption/coco/images/')
parser.add_argument('--annotDir', type=str, default='/users4/zsun/pytorch/image_caption/')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--crop_size', type=int, default=224 ,help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='/users4/zsun/pytorch/image_caption/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--pretrained', type=str, default='', help='start from checkpoint or scratch')
# CNN fine-tuning
parser.add_argument('--fine_tune_start_layer', type=int, default=5,help='CNN fine-tuning layers from: [0-7]')
# Optimizer Adam parameter
parser.add_argument('--cnn_epoch', type=int, default=20, help='start fine-tuning CNN after')
parser.add_argument('--clip', type=float, default=0.1 )
parser.add_argument('--alpha', type=float, default=0.8, help='alpha in Adam' )
parser.add_argument('--beta', type=float, default=0.999, help='beta in Adam' )
parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate for the whole model' )
parser.add_argument('--learning_rate_cnn', type=float, default=1e-5, help='learning rate for fine-tuning CNN' )
parser.add_argument( '--lr_decay', type=int, default=20, help='epoch at which to start lr decay' )
parser.add_argument( '--lr_decay_every', type=int, default=50, help='decay learning rate at every this number')
opt = parser.parse_args()

# set the random seed manually
torch.manual_seed(opt.seed) 
opt.cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda: torch.cuda.manual_seed(opt.seed)

# log config
if not os.path.exists(opt.cache): os.mkdir(opt.cache)
if not os.path.exists(opt.checkpoint): os.mkdir(opt.checkpoint)
if not os.path.exists(opt.save): os.mkdir(opt.save)

# Image preprocessing
transform = transforms.Compose([ # together
    transforms.RandomCrop(opt.crop_size), # random cut 
    transforms.RandomHorizontalFlip(), # horizontal flip
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

with open(opt.vocab_path, 'rb') as f: # opt.vocab = vocab_en
    vocab = pickle.load(f)
    opt.vocab_size = len(vocab)    


base_name = 'info_%s_batch_%d' % (opt.info, opt.batch_size)


LOG_FORMAT = "%(message)s"
log_name = base_name + '.log'
p_log = os.path.join(opt.cache, log_name)#日志输出模块 #以后的info输出在此cache/文件夹
logging.basicConfig(filename=p_log, level=logging.INFO, format=LOG_FORMAT)#文件名日志级别输出格式


# initialize the parameters
# for p in model.parameters(): p.data.uniform_(-0.1, 0.1) 

train_loader, val_loader, test_loader = get_loader(opt, vocab, transform, shuffle=True, 
                                                    num_workers=opt.num_workers) 


adaptive = Encoder2Decoder(opt.ninp, opt.vocab_size, opt.nhid)
if opt.pretrained:  
    # adaptive.load_state_dict( torch.load( opt.pretrained ) )
    adaptive = torch.load( opt.pretrained ) 
if torch.cuda.is_available():
    adaptive.cuda()  

# Constructing CNN parameters for optimization, only fine-tuning higher layers
cnn_subs = list( adaptive.encoder.resnet_conv.children() )[ opt.fine_tune_start_layer: ]
cnn_params = [ list( sub_module.parameters() ) for sub_module in cnn_subs ]
cnn_params = [ item for sublist in cnn_params for item in sublist ]    
# cnn_optimizer = torch.optim.Adam( cnn_params, lr=opt.learning_rate_cnn, betas=( opt.alpha, opt.beta ) )

# create the optimizer
params = list( adaptive.encoder.affine_a.parameters() ) + list( adaptive.encoder.affine_b.parameters() ) \
                + list( adaptive.decoder.parameters() )

# optimizer = getattr(optim, opt.optim)(params, lr=opt.lr, weight_decay=opt.l2) 
# if opt.optim == 'SGD':
#     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opt.lr / (1 + 0.1 * epoch))
 
LMcriterion = nn.CrossEntropyLoss()
train_loss = []    
def train(epoch): 
    adaptive.train() 
    avg_loss = []  
    learning_rate = opt.learning_rate
    learning_rate_cnn = opt.learning_rate_cnn
    if epoch > opt.lr_decay:
        frac = float( epoch - opt.lr_decay ) / opt.lr_decay_every
        decay_factor = math.pow( 0.5, frac )   
        learning_rate = opt.learning_rate * decay_factor
        learning_rate_cnn = opt.learning_rate_cnn * decay_factor
        cnn_optimizer = torch.optim.Adam( cnn_params, lr=learning_rate_cnn, 
                                    betas=( opt.alpha, opt.beta ) )

    optimizer = torch.optim.Adam( params, lr=learning_rate, 
                             betas=( opt.alpha, opt.beta ) )

    for batch_i, (images, captions, lengths) in enumerate(train_loader , start=1):  
         
        start_time = time.time() #images=[batch, 3, 224, 224] # <class 'torch.FloatTensor'>
        src = Variable(images)#torch.LongTensor(images)) # <class 'torch.autograd.variable.Variable'>
        trg = Variable(torch.LongTensor(captions))   
        if torch.cuda.is_available():
            src = src.cuda()
            trg = trg.cuda()
        optimizer.zero_grad() 
        if epoch > opt.cnn_epoch: 
            cnn_optimizer.zero_grad() ######################
        loss = adaptive(src, trg, lengths)   
        loss.backward() 

        # grad_norm = torch.nn.utils.clip_grad_norm(decoder.parameters(), opt.grad_clip)############3        
        for p in adaptive.decoder.LSTMcell.parameters():
            p.data.clamp_( -opt.clip, opt.clip )
        optimizer.step()

        # Start CNN fine-tuning
        if epoch > opt.cnn_epoch: 
            cnn_optimizer.step()
            
        avg_loss.append(loss.data[0])
        elapsed = time.time() - start_time
        if batch_i % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.2e}\tl2: {:.1e}s/batch: {:.2f}'.format(
                    epoch, batch_i, len(train_loader), 100.*batch_i/len(train_loader), loss.data[0], opt.l2, elapsed)
            )
    train_loss.extend(avg_loss)
    return sum(avg_loss) / len(avg_loss)            


def test(epoch, valtest_loader, tag):
    logging.info(time.strftime('time:%m.%d_%H:%M', time.localtime(time.time())))#写在cache下的log文件里
    adaptive.eval()
    prediction = [] 
    for batch_idx, (image, img_id) in enumerate(valtest_loader, 1):  
        # print((image))          #list   #torch.FloatTensor 
        # print((image[0]))       #None   #3x224x224
        if type(image) == tuple:   #list  
            continue   
        start_time = time.time() # image=[1, 3, 224, 224] <class 'torch.FloatTensor'> 
        source = Variable(image) # torch.LongTensor(image))#transform(image).unsqueeze(0))      
        if torch.cuda.is_available():  
            src = source.cuda()       
        pred = adaptive.sampler(opt, src)   # print(pred.size()) [1, 20] 
        # pred = adaptive.beam(opt, src)   # print(pred.size()) [1, 20] 
        pred_dict = dict()  # print(len(vocab)) 9956
        pred = [' '.join(map(lambda x: vocab.idx2word[x], p)) for p in pred]
        pred_dict['image_id'] = img_id[0]
        pred_dict['caption'] = pred[0] 
        prediction.append(pred_dict)  

        elapsed = time.time() - start_time
        if batch_idx % 1000 == 0:#opt.log_interval
            print(tag + ' Epoch: [{}/{} ({:.0f}%)]\ts/batch: {:.2f}'.format(
                batch_idx, len(valtest_loader), 100. * batch_idx / len(valtest_loader), elapsed) )

    name = ('epoch_%d_%s' % (epoch, tag)) + base_name # test/val生成的文件，放在generation下 
    
    predictfile = '%s%s'%(opt.save, name)
    json.dump(prediction, open(predictfile+'.json', 'w')) 
    metrics = get_metrics(tag, predictfile)
    logging.info("%s epoch %s metrics %s" %(tag, str(epoch), str(metrics)))
    print("%s epoch %s metrics %s" %(tag, str(epoch), str(metrics)))

for epoch in range(opt.epoch, opt.nepoch): #xrange： 
    train_avg_loss = train(epoch) # 6471
    test(epoch, val_loader, 'val') # 202654
    test(epoch, test_loader, 'test') # 202654
    name = base_name + 'train_%.2e_e_%d_statedict.pt' % (train_avg_loss, epoch) 
    # state_dict_en = adaptive.state_dict() #checkpoint目录下存放保存的模型参数 
    # torch.save(state_dict_en, os.path.join(opt.checkpoint, name)) 
    torch.save(adaptive, os.path.join(opt.checkpoint, name))   