import torch


class Beam(object):
    def __init__(self, state, beam_size, max_len, dec_eos, dec_pad, is_cuda):
        self.beam_size = beam_size
        self.max_len = max_len
        self.dec_eos = dec_eos
        self.dec_pad = dec_pad
        self.stop = False

        self.pointer = []
        self.pred = [torch.LongTensor(self.beam_size).fill_(self.dec_eos)]
        self.score = torch.FloatTensor(self.beam_size).zero_()
        self.state = state

        if is_cuda:
            self.pred[0] = self.pred[0].cuda()
            self.score = self.score.cuda()

    def extract_input(self):
        return self.pred[-1]

    def extract_state(self):
        return self.state

    def extract_ptr(self):
        return self.pointer[-1]

    def step(self, prob):
        if len(self.pointer) > 0:
            cur_prob = prob + self.score.unsqueeze(1).expand_as(prob)
        else:
            cur_prob = prob[0]
        self.score, ix = cur_prob.view(-1).topk(self.beam_size)
        self.pointer.append(ix / prob.size(1))
        self.pred.append(ix - self.pointer[-1] * prob.size(1))

        if self.pred[-1][0] == self.dec_eos or len(self.pred) >= self.max_len:
            self.stop = True

    def top_1(self):
        score, ix = torch.sort(self.score, 0, True)
        return score[0], ix[0]

    def get_hyp(self, k):
        hyp = []
        for i in range(len(self.pointer) - 1, -1, -1):
            hyp.append(self.pred[i+1][k])
            k = self.pointer[i][k]
        hyp = hyp[::-1]
        while hyp[-1] == self.dec_pad:
            hyp.pop()
        if hyp[-1] == self.dec_eos:
            hyp.pop()
        return hyp
