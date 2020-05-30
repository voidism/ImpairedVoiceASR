import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import pdb
from transformers import AdamW

class BopomoTokenizer(object):
    def __init__(self, vocabulary='bopomo_char_file'):
        self.vocabulary = [x.strip() for x in open(vocabulary).readlines()]
        self.vocab_dict = {x: i for i, x in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary)
        
    def token2id(self, token):
        if token in self.vocab_dict:
            return self.vocab_dict[token]
        else:
            return self.vocab_dict['<unk>']

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_tokens_in_words(x) for x in tokens]

    def get_type(self, token):
        if token in range(13, 34):
            return 1
        elif token in range(47, 50):
            return 2
        elif token in range(34, 47):
            return 3
        elif token in range(8, 13):
            return 4
        else:
            return 0

    def convert_tokens_in_words(self, tokens):
        ids = [self.token2id(x) for x in tokens]
        types = [self.get_type(i) for i in ids]
        if types == [1, 2, 3, 4]:
            return ids
        elif types == [2, 3, 4]:
            return [self.vocab_dict['<null_con>']] + ids
        elif types == [1, 3, 4]:
            return [ids[0], self.vocab_dict['<null_mid>']] + ids[1:]
        elif types == [1, 2, 4]:
            return ids[:2] + [self.vocab_dict['<null_vow>'], ids[2]]
        elif types == [1, 4]:
            return [ids[0], self.vocab_dict['<null_mid>'], self.vocab_dict['<null_vow>'], ids[1]]
        elif types == [2, 4]:
            return [self.vocab_dict['<null_con>'], ids[0], self.vocab_dict['<null_vow>'], ids[1]]
        elif types == [3, 4]:
            return [self.vocab_dict['<null_con>'], self.vocab_dict['<null_mid>'], ids[0], ids[1]]
        else:
            print(f"Warning: invalid token {tokens}")
            return (ids+[self.vocab_dict['<unk>']]*4)[:4]

    # def convert_sents_to_tensors(self, sents):
    #     ret = []
    #     for i in range(len(sents))

    def id2token(self, id):
        return self.vocabulary[id]

    def convert_ids_to_tokens(self, ids):
        return [self.id2token(x) for x in ids]

class TextDataset(Dataset):
    def __init__(self, filename, vocabulary='bopomo_char_file', raw=False):
        super(TextDataset, self).__init__()
        self._data = []
        self.bpmf_tokenizer = BopomoTokenizer(vocabulary)
        self.word_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
        self.raw = raw
        for line in tqdm.tqdm(open(filename).readlines()):
            line = line.strip().split(' ')
            inputs = []
            outputs = []
            for term in line:
                w, b = term.split(':')
                inputs.append(b)
                outputs.append(w)
            self._data.append((inputs, outputs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if self.raw:
            return self._data[index]
        else:
            bids = self.bpmf_tokenizer.convert_tokens_to_ids(self._data[index][0])
            wids = self.word_tokenizer.convert_tokens_to_ids(self._data[index][1])
            return torch.tensor(bids), torch.tensor(wids)

    @classmethod
    def collate_fn(self, batch):
        target = [x[1] for x in batch]
        inputs = [x[0] for x in batch]
        max_len = max([len(x) for x in target])
        batch_inputs = []
        batch_target = []
        for i in range(len(inputs)):
            if len(inputs[i]) == max_len:
                batch_inputs.append(inputs[i])
                batch_target.append(target[i])
            else:
                batch_inputs.append(torch.cat([inputs[i], torch.zeros(max_len - len(inputs[i]), 4, dtype=torch.int64)], dim=0))
                batch_target.append(torch.cat([target[i], torch.zeros(max_len - len(target[i]), dtype=torch.int64)], dim=0))
        return torch.stack(batch_inputs).cuda(), torch.stack(batch_target).cuda()


# model
class BopomoEmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, scale=4):
        super(BopomoEmbeddingModel, self).__init__()
        self.bpm_emb = torch.nn.Embedding(vocab_size, hidden_size)
        hidden_size = hidden_size * scale
        self.prenet1 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU(hidden_size)
        self.prenet2 = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.bpm_emb(x).view(x.size(0), x.size(1), -1)
        return self.prenet2(self.relu(self.prenet1(x)))
#if __name__ == '__main__':
    #train_dataset = TextDataset('data-aishell-train.txt')
    #dev_dataset = TextDataset('data-aishell-dev.txt')
bpmf_tokenizer = BopomoTokenizer('bopomo_char_file')
word_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
vocab_size = bpmf_tokenizer.vocab_size
hidden_size = 768 // 4
bpm_model = BopomoEmbeddingModel(vocab_size, hidden_size, scale=4)
bpm_model.cuda()
bpm_model.load_state_dict(torch.load("bpm_model.ckpt"))
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
model = transformers.BertForMaskedLM.from_pretrained('./cbert')
model.cuda()
model.load_state_dict(torch.load("model.ckpt"))
#while True:
#    inputs = input("BOPOMOFO>>> ").split(' ')
    #inputs = "ㄓㄜˋ ㄐㄧㄢˋ ㄕˋ ㄑㄧㄥˊ ㄐㄧㄡˋ ㄓㄜˋ ㄧㄤˋ ㄅㄚ˙".split(' ')
def z2c(inputs):
    inputs = torch.tensor(bpmf_tokenizer.convert_tokens_to_ids(inputs)).unsqueeze(0).cuda()
    mask = torch.ones(inputs.shape[0], inputs.shape[1], dtype=torch.uint8).cuda()
    allinputs = {"inputs_embeds": bpm_model(inputs), "attention_mask": mask}
    outputs = model(**allinputs)[0]
    pred = outputs.argmax(-1).tolist()[0]
    return ''.join(word_tokenizer.convert_ids_to_tokens(pred))
'''
if False:
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=TextDataset.collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=TextDataset.collate_fn)
# for b in d:
    # break

# train_dataset = SentiDataset("train_split.txt", tokenizer)
# dev_dataset = SentiDataset("dev_split.txt", tokenizer)
# train_sampler = RandomSampler(train_dataset)
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64, collate_fn=SentiDataset.collate_fn)
# dev_dataloader = DataLoader(dev_dataset, batch_size=32, collate_fn=SentiDataset.collate_fn)
# model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base').cuda()
n_epoch = 100
t_total = len(train_dataloader) * n_epoch
optimizer = AdamW([{'params':model.parameters(), 'lr':5e-5}, {'params':bpm_model.parameters(), 'lr':5e-4}], eps=1e-8)
#scheduler = get_linear_schedule_with_warmup(
#    optimizer, num_warmup_steps=100, num_training_steps=t_total
#)
# model.load_state_dict(torch.load('premodel.ckpt'))
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
best_acc = 0.0
for epoch in range(n_epoch):
    cur = 0
    tot = 0
    model.train()
    for step, (batch, labels) in enumerate(train_dataloader):
        if epoch == 0:
            break
        inputs = {"inputs_embeds": bpm_model(batch.cuda()), "attention_mask": (labels!=0).cuda()}
        outputs = model(**inputs)
        vocab_size = outputs[0].shape[-1]
        loss = loss_fn(outputs[0].view(-1, vocab_size), labels.view(-1).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(bpm_model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
        model.zero_grad()
        if step % 10 == 0:
            cur += int(((outputs[0].argmax(-1).detach() == labels)*(labels!=0)).cpu().sum())
            tot += int((labels!=0).cpu().sum())
            print('epoch: %d step: %d loss: %4f acc: %4f'%(epoch, step, float(loss), float(cur)/float(tot)), flush=True)
    model.eval()
    cur = 0
    tot = 0
    for step, (batch, labels) in enumerate(dev_dataloader):
        inputs = {"inputs_embeds": bpm_model(batch.cuda()), "attention_mask": (labels!=0).cuda()}
        outputs = model(**inputs)
        vocab_size = outputs[0].shape[-1]
        loss = loss_fn(outputs[0].view(-1, vocab_size), labels.cuda().view(-1)).detach().cpu()
        cur += int(((outputs[0].argmax(-1).detach() == labels)*(labels!=0)).cpu().sum())
        tot += int((labels!=0).cpu().sum())
        print('epoch: %d dev-step: %d loss: %4f acc: %4f'%(epoch, step, float(loss), float(cur)/float(tot)), flush=True)
    acc = float(cur)/float(tot)
    if acc>best_acc:
        print('save model to model.ckpt, acc: %4f > %4f'%(acc, best_acc), flush=True)
        torch.save(model.state_dict(), 'model.ckpt')
        torch.save(bpm_model.state_dict(), 'bpm_model.ckpt')
        best_acc = acc
'''
