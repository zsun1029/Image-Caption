import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
from beam import Beam
        
class AttentiveCNN( nn.Module ):
    def __init__( self, embed_size, hidden_size ):
        super( AttentiveCNN, self ).__init__()
        resnet = models.resnet152( pretrained=True )
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature 
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d( 7 )#//// # kernel_size = 7
        self.affine_a = nn.Linear( 2048, hidden_size ) # v_i = W_a * A
        self.affine_b = nn.Linear( 2048, embed_size )  # v_g = W_b * a^g
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_a.weight, mode='fan_in' )#//////////
        init.kaiming_uniform( self.affine_b.weight, mode='fan_in' )#/////
        self.affine_a.bias.data.fill_( 0 )
        self.affine_b.bias.data.fill_( 0 )
        
    def forward( self, images ):  
        A = self.resnet_conv( images ) #//////# [batch, 2048, 7, 7]
        
        a_g = self.avgpool( A ) # [batch, 2048, 7, 7]
        a_g = a_g.view( a_g.size(0), -1 ) # a_g=[batch, 2048]
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )#1,2维转置 #V[batch,49,2048]
        V = F.relu( self.affine_a( self.dropout( V ) ) )     # V[batch,49,nhid]        
        v_g = F.relu( self.affine_b( self.dropout( a_g ) ) ) # v_g[batch,ninp]        
        return V, v_g               

# Attention Block for C_hat calculation
class Atten( nn.Module ):
    def __init__( self, hidden_size ):
        super( Atten, self ).__init__() 
        self.affine_v = nn.Linear( hidden_size, 49, bias=False ) # W_v
        self.affine_g = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_s = nn.Linear( hidden_size, 49, bias=False ) # W_s
        self.affine_h = nn.Linear( 49, 1, bias=False ) # w_h 
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ): 
        init.xavier_uniform( self.affine_v.weight )
        init.xavier_uniform( self.affine_g.weight )
        init.xavier_uniform( self.affine_h.weight )
        init.xavier_uniform( self.affine_s.weight )
        
    def forward( self, V, h_t ):
        # Input: V=[v_1, v_2, ... v_k][8, 49, 1024], h_t=s_t [8, 1024] from LSTM
        # W_v * V[8,49,49] + W_g * h_t * 1^T[8,49,1] 
        content_v = self.affine_v( self.dropout(V) ) \
                    + self.affine_g( self.dropout(h_t) ).unsqueeze(2) # content_v [8,49,49]
        # z_t = W_h * tanh( content_v )      
        z_t = self.affine_h( self.dropout( F.tanh( content_v ) ) ).squeeze(2)  #[8,49,1]->[8,49]
        alpha_t = F.softmax(z_t.view(-1, z_t.size(1)), 1) # [8,49]
        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm( alpha_t.unsqueeze(1), V ).squeeze(1) #c_t=[B=8, 1, 1024]->[8,1024] 
        return c_t, alpha_t


# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding    
class AdaptiveBlock( nn.Module ): 
    def __init__( self, embed_size, hidden_size, vocab_size ):
        super( AdaptiveBlock, self ).__init__() 
        # self.sentinel = Sentinel( embed_size * 2, hidden_size ) 
        self.atten = Atten( hidden_size ) 
        self.mlp = nn.Linear( hidden_size, vocab_size ) 
        self.dropout = nn.Dropout( 0.5 ) # Dropout layer inside Affine Transformation
        self.hidden_size = hidden_size
        self.init_weights()
        
    def init_weights( self ):
        # Initialize final classifier weights
        init.kaiming_normal( self.mlp.weight, mode='fan_in' )
        self.mlp.bias.data.fill_( 0 )
                
    def forward( self, hiddens, V ): # x_t=hiddens=cells[batch,ninp]   
        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights = self.atten( V, hiddens ) 
        # Final score along vocabulary
        scores = self.mlp(self.dropout(c_hat + hiddens))
        return scores, atten_weights
    
    def init_hidden( self, bsz ):
        # Hidden_0 & Cell_0 initialization 
        weight = next( self.parameters() ).data 
        if torch.cuda.is_available():
            return ( Variable( weight.new( 1 , bsz, self.hidden_size ).zero_().cuda() ),
                    Variable( weight.new( 1,  bsz, self.hidden_size ).zero_().cuda() ) ) 
        else: 
            return ( Variable( weight.new( 1 , bsz, self.hidden_size ).zero_() ),
                    Variable( weight.new( 1,  bsz, self.hidden_size ).zero_() ) ) 
    

class Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Decoder, self ).__init__() 
        self.embed = nn.Embedding( vocab_size, embed_size ) 
        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTMcell = nn.LSTMCell( embed_size * 2, hidden_size )
        self.hidden_size = hidden_size 
        self.adaptive = AdaptiveBlock( embed_size, hidden_size, vocab_size )
        
    def forward( self, V, v_g , captions, h_t_1, states): 
        embeddings = self.embed( captions ) # w_t=[batch, len, ninp]
        # print(captions.size()) #[8]         embeddings=[8, 512]
        # x_t = [w_t;v_g]                   # v_g=[batch,ninp]
        # print(embeddings.size()) #[1,1,512] 
        x_t = torch.cat( ( embeddings, v_g ), dim=1 ) 
        # print(x_t.size())[8, 1024] [batch,ninp] 
        # print(h_t.size())[8, 1024]
        # print(states.size()) [8, 1024]
        h_t, states = self.LSTMcell( x_t, (h_t_1, states) )  

        scores, atten_weights = self.adaptive( h_t, V ) 
        return h_t, states, scores
     

# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__() 
        self.encoder = AttentiveCNN( embed_size, hidden_size )
        self.decoder = Decoder( embed_size, vocab_size, hidden_size )
        self.hidden_size = hidden_size
    def forward( self, images, captions, lengths ):  # print(lengths) 108 
        V, v_g = self.encoder( images ) # V=[ v_1, ..., v_k ], v_g  
        # Hiddens: [Batch, seq_len, hidden_size]
        # Cells: [seq_len, Batch, hidden_size], default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = Variable( torch.zeros( captions.size(0), self.hidden_size ).cuda() )
            cells = Variable( torch.zeros( captions.size(0), self.hidden_size ).cuda() )
        else:
            hiddens = Variable( torch.zeros( captions.size(0), self.hidden_size ) )
            cells = Variable( torch.zeros( captions.size(0), self.hidden_size ) ) 
        
        loss = 0 
        captions = captions.t() # [batch, len] -> [len, batch] 
        # pack_scores = Variable( torch.zeros( captions.size(1), 1, 9956).cuda() ) 
        for i in range(captions.size(0)- 1):  # 
            hiddens, cells, scores = self.decoder( V, v_g, captions[i], hiddens, cells ) 
        #     # print(scores.size()) #[8, 9956] [batch, *]
        #     pack_scores = torch.cat((pack_scores, scores.unsqueeze(1)), dim=1) 
        # # Pack it to make criterion calculation more efficient 
        # packed_scores = pack_padded_sequence( pack_scores[:,1:,:], lengths, batch_first=True ) 
            loss += F.cross_entropy(scores, captions[i + 1])
            print(type(loss))
            print(type(loss.data))
        return loss # packed_scores
    
    # Caption generator
    def sampler( self, opt, images, max_len=20 ): 
        V, v_g = self.encoder( images ) 
        if torch.cuda.is_available():
            sos = Variable( torch.LongTensor( images.size( 0 ) ).fill_( 1 ).cuda() ) 
        else:
            sos = Variable( torch.LongTensor( images.size( 0 )).fill_( 1 ) )
         
        # Initial hidden states
        if torch.cuda.is_available():
            hiddens = Variable( torch.zeros( images.size(0), self.hidden_size ).cuda() )
            cells = Variable( torch.zeros( images.size(0), self.hidden_size ).cuda() )
        else:
            hiddens = Variable( torch.zeros( images.size(0), self.hidden_size ) )
            cells = Variable( torch.zeros( images.size(0), self.hidden_size ) ) 

        pred = [[] for _ in range(images.size( 0 ))] 
        status = V.data.new(images.size( 0 )).fill_(1)
        input = sos
        # print(input.size())[1]
        while status.sum() != 0 and len(pred[0])<=max_len:
            hiddens, cells, scores = self.decoder( V, v_g, input, hiddens, cells ) 
            # print(scores) #[1, 9956].size()
            # print(F.softmax(scores)) #[1, 9956].size()
            input = F.softmax(scores, dim=1).max(1)[ 1 ] # argmax   
            # print(input.size()) [1]
            for k, v in enumerate(input.data.squeeze()):  
                if v == opt.dec_eos or len(pred[k]) > max_len:
                    status[k] = 0
                if v != opt.dec_eos and status[k] == 1:
                    pred[k].append(v)   
        return pred 


    def beam(self, images, beam_size=3, max_len=20):
        # context = self.encoder(images)
        V, v_g = self.encoder( images ) 

        batch_size = images.size(0)

        # Initial hidden states
        # hiddens = F.tanh(self.init_affine(context.mean(1)))
        if torch.cuda.is_available():
            hiddens = Variable( torch.zeros( batch_size, self.hidden_size ).cuda() )
            cells = Variable( torch.zeros( batch_size, self.hidden_size ).cuda() )
        else:
            hiddens = Variable( torch.zeros( batch_size, self.hidden_size ) )
            cells = Variable( torch.zeros( batch_size, self.hidden_size ) ) 

        PAD = Variable(images.data.new(1).fill_(opt.dec_pad))

        # sequence = [
        #     Beam(
        #         [hiddens[i].unsqueeze(0).repeat(beam_size, 1), context[i].unsqueeze(0).repeat(beam_size, 1, 1)],
        #         beam_size, max_len, opt.dec_eos, opt.dec_pad, context.is_cuda)
        #     for i in range(batch_size)
        # ]
        sequence = [
            Beam(
                [hiddens[i].unsqueeze(0).repeat(beam_size, 1), cells[i].unsqueeze(0).repeat(beam_size, 1, 1)],
                beam_size, max_len, opt.dec_eos, opt.dec_pad, v_g.is_cuda)
            for i in range(batch_size)
        ]
        print(hiddens.size())
        print(cells.size())
        status = images.data.new(batch_size).fill_(1)
        step = 0
        while status.sum() != 0 and step < max_len:
            input = torch.cat([seq.extract_input() for seq in sequence if not seq.stop], 0)
            input = Variable(input, volatile=True)
            hiddens = torch.cat([seq.extract_state()[0] for seq in sequence if not seq.stop], 0)
            cells = torch.cat([seq.extract_state()[1] for seq in sequence if not seq.stop], 0)            
            print(hiddens.size())
            print(cells.size())
            # output, hiddens = self.decoder(self.emb(input), hiddens, context)
            hiddens, cells, scores = self.decoder( V, v_g, input, hiddens, cells ) 
            # prob = F.softmax(self.affine(self.dropout(output)), dim=1)
            prob = F.softmax(scores, dim=1)
            prob = prob.index_fill(1, PAD, 0)
            prob = prob.view(-1, beam_size, prob.size(-1))
            batch_ix_map = status.cumsum(dim=0).add(-1)
            hiddens = hiddens.view(-1, beam_size, self.hidden_size)
            for i in range(batch_size):
                if status[i] == 0:
                    continue
                batch_ix = batch_ix_map[i]
                sequence[i].step(prob.data.select(0, batch_ix))
                if sequence[i].stop:
                    status[i] = 0
                beam_hidden = hiddens.data.select(0, batch_ix)
                beam_hidden.copy_(beam_hidden.index_select(0, sequence[i].extract_ptr()))
                sequence[i].state[0] = hiddens[batch_ix]
            step += 1
        pred = []
        for i in range(batch_size):
            top_p, top_x = sequence[i].top_1()
            pred.append(sequence[i].get_hyp(top_x))
        return pred