import os
import torchvision.datasets as dset
# import torchvision.transforms as transforms
# cap = dset.CocoCaptions(root='/users4/zsun/pytorch/image_caption/coco/images/train2014', 
#                             annFile='/users4/zsun/pytorch/image_caption/coco/annotations/captions_train2014.json', 
#                             transform=transforms.ToTensor()) # transform
# img, target = cap[0]
# # print("Image Size: ", img.size()) # (3L, 427L, 640L) [3, 480, 640]#<class 'torch.FloatTensor'>
import nltk
from PIL import Image
import torch.utils.data as data
import torch
from pycocotools.coco import COCO
import subprocess 

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""    
    def __init__(self, opt, vocab, imagef, annotf=None, transform=None):  
        root = os.path.join(opt.imageDir, imagef)
        json = os.path.join(opt.annotDir, annotf) 
        opt.dec_pad = vocab.word2idx['<pad>']
        opt.dec_eos = vocab.word2idx['<end>']  # EOS token            
        opt.dec_sos = vocab.word2idx['<sos>']  # sos token            
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys()) 
        self.vocab = vocab
        self.transform = transform
    
    def __getitem__(self, index):
        """Returns one data pair (image and caption).""" 
        # print(index)
        coco = self.coco
        vocab = self.vocab 
        ann_id = self.ids[index] 
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB') 
        if self.transform is not None:
            image = self.transform(image) 
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<sos>'))
        caption.extend([vocab(tok) for tok in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption) 
        return image, target

    def __len__(self):
        return len(self.ids) 

class CocoevalDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""    
    def __init__(self, opt, vocab, imagef, annotf=None, transform=None):  
        root = os.path.join(opt.imageDir, imagef)
        json = os.path.join(opt.annotDir, annotf)                   
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())        
        # self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        namelist = []          
        for filename in os.listdir(self.root):
            namelist.append(filename)        
        namelist.sort()
        self.namelist = namelist
    def __getitem__(self, index):
        """Returns one data pair (image and caption).""" 
        coco = self.coco
        vocab = self.vocab
        # ann_id = self.ids[index] 
        # img_id = coco.anns[ann_id]['image_id']
        img_id = self.ids[index] 
        path = coco.loadImgs(img_id)[0]['file_name']   
        if path not in self.namelist:
            return 'None', 0 
        image = Image.open(os.path.join(self.root, path)).convert('RGB') 
        if self.transform is not None:
            image = self.transform(image) 
        return image, img_id

    def __len__(self):
        return len(self.ids) 

def collate_fn(data): 
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True) # len(x[1]) = len(caption)
    # print(len(data))    64 tuple
    # print(data[0][0].size())  [3,224,224]
    # print(data[0][1].size())  [len]
    images, captions = zip(*data) # len(images))    64//images[0].size())    [3,224,224]
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]  
    targets = torch.zeros(len(captions), max(lengths)).long() # [410000, ]
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end] # the rest of them ars zeros 
    # print(images.size())  #train[64, 3, 224, 224]  val[1, 3, 224, 224]
    # print(targets.size())      #[64, len]             [1, 12]
    # print(len(lengths))        # 64                    1
    return images, targets, lengths
def collate_fn_eval(data):  
    # print(type(data))      #batch=1 list 
    # print(len(data))      #batch=1 list 
    # print(type(data[0]))      # tuple
    # print(data[0][0] )    #[3,224,224]  data[0][0]=str
    # print(data[0][1] )    #[len]         0
    images, img_id = zip(*data)     # list,  len=batch=1, images[0]=[3,224,224]
    # images, img_id = data[0]     # list,  len=batch=1, images[0]=[3,224,224]
    # print(type(img_id)) # tuple
    # print((img_id))     # (0, )
    if images[0] is not 'None':
        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)  
    return images, img_id #[1, 3, 224, 224] or [str='None']

def get_loader(opt, vocab, transform, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    
    coco_train = CocoDataset(opt, vocab=vocab, imagef='resizedtrain2014', 
                            annotf='coco/annotations/captions_train2014.json', transform=transform)
    coco_val = CocoevalDataset(opt, vocab=vocab, imagef='resize224val2014val', 
                            annotf='coco-caption/annotations/captions_val2014.json', transform=transform)
    coco_test = CocoevalDataset(opt, vocab=vocab, imagef='resize224val2014test', 
                            annotf='coco-caption/annotations/captions_val2014.json', transform=transform)

    # Data loader for COCO dataset will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size). 
    train_loader = torch.utils.data.DataLoader(dataset=coco_train,  batch_size=opt.batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=coco_val,  batch_size=opt.test_batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_eval)
    test_loader = torch.utils.data.DataLoader(dataset=coco_test,  batch_size=opt.test_batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_eval)

    # print(len(train_loader)) # 6471 = 82783*5/64
    # print(len(val_loader)) # 202654 = 40531*5/1
    # print(len(test_loader)) # 0 
    return train_loader, val_loader, test_loader # test_loader train_loader, train_loader #


def get_metrics(tag, predictfile):
    myevaldir = "/users4/zsun/pytorch/image_caption/coco-caption"   
    # print(type(predict))     #list
    # print(len(predict))     # 101328=20664*5
    # print(type(predict[0])) #tuple
    # print((predict[0]))     #(70434, ) 
    outputs = subprocess.check_output("source activate py27\npython2 %s/myeval.py \
                            --pred %s"%(myevaldir, predictfile), shell=True)  
    return outputs[-98:]
