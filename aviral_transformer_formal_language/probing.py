from torch.utils.data import Dataset, DataLoader

def pad_seq(seq, max_length, voc):
	seq += [voc.get_id('T') for i in range(max_length - len(seq))]
	return seq

def sent_to_idx(voc, sent, max_length=-1):
	idx_vec = []
	for w in sent:
        idx = voc.get_id(w)
        idx_vec.append(idx)
        
	idx_vec.append(voc.get_id('T'))
	idx_vec= pad_seq(idx_vec, max_length+1, voc)
	return idx_vec

def sents_to_idx(voc, sents):
	max_length = max([len(s) for s in sents])
	all_indexes = []
	for sent in sents:
		all_indexes.append(sent_to_idx(voc, sent, max_length))

	all_indexes = torch.tensor(all_indexes, dtype= torch.long)
	return all_indexes

class ProbingDataset(Dataset):
    def __init__(self, data, voc, model):
        self.raw = data
        self.voc = voc
        
        self.__tokenise__()
        
    def __tokenise__(self):
        data_ids = sents_to_idx(self.voc, self.raw)
        self.data = data_ids[:, :-1].transpose(0, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]