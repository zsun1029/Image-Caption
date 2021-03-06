import argparse
import time
import torch
from torch.autograd import Variable
import os
# from modelgru import EncoderCNN, DecoderRNN
from PIL import Image
import pickle
from build_vocab import Vocabulary
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Sequence-to-Tree model')
parser.add_argument('--imageDir', type=str, default='/users4/zsun/pytorch/image_caption/coco/images/resize224test2014/')#
parser.add_argument('--imagename', type=str, default='COCO_test2014_000000000080.jpg')#
#COCO_test2014_000000000001#90
parser.add_argument('--checkpoint', type=str, default='./checkpoint/', help='path to save the model')
parser.add_argument('--model', type=str, default='info_3_batch_16train_2.59e+01_e_49_en.pt')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--ninp', type=int, default=512, help='size of word embedding')
parser.add_argument('--nhid', type=int, default=1024, help='number of hidden layer')
parser.add_argument('--vocab_path', type=str, default='/users4/zsun/pytorch/image_caption/vocab.pkl')
# parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--crop_size', type=int, default=224 ,help='size for randomly cropping images')
parser.add_argument('--seed', type=int, default=123, help='random number seed')
parser.add_argument('--pred_max_len', type=int, default=200, help='maximum length of generated sequence')
opt = parser.parse_args()

opt.cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda: torch.cuda.manual_seed(opt.seed)
with open(opt.vocab_path, 'rb') as f:  
    vocab = pickle.load(f)
    opt.vocab_size = len(vocab)  
opt.dec_pad = vocab.word2idx['<pad>']
opt.dec_eos = vocab.word2idx['<end>']  # EOS token            
opt.dec_sos = vocab.word2idx['<sos>']  # sos token   
 
model_path = os.path.join(opt.checkpoint, opt.model) 
model = torch.load(model_path)
if torch.cuda.is_available():
    model.cuda() 


def test(image):
    start_time = time.time() # image=[1, 3, 224, 224] <class 'torch.FloatTensor'> 
    source = Variable(image) # torch.LongTensor(image))#transform(image).unsqueeze(0))      
    if torch.cuda.is_available():  
        src = source.cuda()    
    src = src.unsqueeze(0)
    pred = model.sampler(opt, src)    
    pred = [' '.join(map(lambda x: vocab.idx2word[x], p)) for p in pred]   
    elapsed = time.time() - start_time 
    print('%.2f s: %s'%(elapsed, pred)) 

transform = transforms.Compose([ # together
    # transforms.RandomCrop(opt.crop_size), # random cut 
    transforms.RandomHorizontalFlip(), # horizontal flip
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# imagenamelist = ['COCO_test2014_000000000106.jpg','COCO_test2014_000000000108.jpg','COCO_test2014_000000000128.jpg','COCO_test2014_000000000155.jpg','COCO_test2014_000000000171.jpg','COCO_test2014_000000000173.jpg','COCO_test2014_000000000178.jpg',\
# 'COCO_test2014_000000000180.jpg','COCO_test2014_000000000182.jpg','COCO_test2014_000000000184.jpg','COCO_test2014_000000000188.jpg','COCO_test2014_000000000191.jpg']
imagenamelist = ['COCO_test2014_000000000202.jpg','COCO_test2014_000000000212.jpg','COCO_test2014_000000000219.jpg','COCO_test2014_000000000229.jpg','COCO_test2014_000000000245.jpg','COCO_test2014_000000000251.jpg','COCO_test2014_000000000275.jpg','COCO_test2014_000000000276.jpg']
for imagename in imagenamelist:
    for i in range(5):
        image = Image.open(os.path.join(opt.imageDir, imagename)).convert('RGB') 
        image = transform(image)  
        # print(image.size())
        test(image) # 202654



