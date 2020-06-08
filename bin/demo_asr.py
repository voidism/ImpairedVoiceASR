import os
import copy
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed
from torch.utils.tensorboard import SummaryWriter

from src.solver import BaseSolver
from src.asr import ASR
from src.decode import BeamDecoder
from src.optim import Optimizer
from src.data import load_dataset, load_text_encoder
from src.online import Datadealer
from src.util import human_format, cal_er, feat_to_fig, Timer
import pdb

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)

        # ToDo : support tr/eval on different corpus
        assert self.config['data']['corpus']['name'] == self.src_config['data']['corpus']['name']
        self.config['data']['corpus']['path'] = self.src_config['data']['corpus']['path']
        self.config['data']['corpus']['bucketing'] = False

        # The follow attribute should be identical to training config
        self.config['data']['audio'] = self.src_config['data']['audio']
        self.config['data']['corpus']['train_split'] = self.src_config['data']['corpus']['train_split']
        self.config['data']['text'] = self.src_config['data']['text']
        self.tokenizer = load_text_encoder(**self.config['data']['text'])
        self.config['model'] = self.src_config['model']
        self.finetune_first = 5
        self.best_wer = {'att': 3.0, 'ctc': 3.0}

        # Output file
        self.output_file = str(self.ckpdir)+'_{}_{}.csv'

        # Override batch size for beam decoding
        self.greedy = self.config['decode']['beam_size'] == 1
        self.dealer = Datadealer(self.config['data']['audio'])
        self.ctc = self.config['decode']['ctc_weight'] == 1.0
        if not self.greedy:
            self.config['data']['corpus']['batch_size'] = 1
        else:
            # ToDo : implement greedy
            raise NotImplementedError

        # Logger settings
        self.logdir = os.path.join(paras.logdir, self.exp_name)
        self.log = SummaryWriter(
            self.logdir, flush_secs=self.TB_FLUSH_FREQ)
        self.timer = Timer()

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt != 0, dim=-1)

        return feat, feat_len, txt, txt_len

    def load_data(self, batch_size=7):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        prev_batch_size = self.config['data']['corpus']['batch_size']
        self.config['data']['corpus']['batch_size'] = batch_size
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.paras.njobs, self.paras.gpu,
                         self.paras.pin_memory, False, **self.config['data'])
        self.config['data']['corpus']['batch_size'] = prev_batch_size
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        self.feat_dim = 120
        self.vocab_size = 46 
        init_adadelta = True
        ''' Setup ASR model and optimizer '''
        # Model
        # init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'
        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                         self.src_config['model']).to(self.device)
        self.verbose(self.model.create_msg())

        if self.finetune_first>0:
            names = ["encoder.layers.%d"%i for i in range(self.finetune_first)]
            model_paras = [{"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in names)]}]
        else:
            model_paras = [{'params': self.model.parameters()}]

        # Losses
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)

        # Plug-ins
        self.emb_fuse = False
        self.emb_reg = ('emb' in self.config) and (
            self.config['emb']['enable'])
        if self.emb_reg:
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb']).to(self.device)
            model_paras.append({'params': self.emb_decoder.parameters()})
            self.emb_fuse = self.emb_decoder.apply_fuse
            if self.emb_fuse:
                self.seq_loss = torch.nn.NLLLoss(ignore_index=0)
            self.verbose(self.emb_decoder.create_msg())

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.src_config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()
        # Beam decoder
        self.decoder = BeamDecoder(
            self.model, self.emb_decoder, **self.config['decode'])
        self.verbose(self.decoder.create_msg())
        # del self.model
        # del self.emb_decoder
        self.decoder.to(self.device)

    def exec(self):
        ''' Testing End-to-end ASR system '''
        while True:
            try:
                filename = input("Input wav file name: ")
                if filename == "exit":
                    return
                feat, feat_len = self.dealer(filename)
                feat = feat.to(self.device)
                feat_len = feat_len.to(self.device)
                # Decode
                with torch.no_grad():
                    hyps = self.decoder(feat, feat_len)

                hyp_seqs = [hyp.outIndex for hyp in hyps]
                hyp_txts = [self.tokenizer.decode(hyp, ignore_repeat=self.ctc) for hyp in hyp_seqs]
                for txt in hyp_txts:
                    print(txt)
            except:
                print("Invalid file")
                pass

    def recognize(self, filename):
        try:
            feat, feat_len = self.dealer(filename)
            feat = feat.to(self.device)
            feat_len = feat_len.to(self.device)
            # Decode
            with torch.no_grad():
                hyps = self.decoder(feat, feat_len)
            
            hyp_seqs = [hyp.outIndex for hyp in hyps]
            hyp_txts = [self.tokenizer.decode(hyp, ignore_repeat=self.ctc) for hyp in hyp_seqs]
            return hyp_txts[0]
        except Exception as e:
            print(e)
            app.logger.debug(e)
            return "Invalid file"

    def fetch_finetune_data(self, filename, fixed_text):
        feat, feat_len = self.dealer(filename)
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        text = self.tokenizer.encode(fixed_text)
        text = torch.tensor(text).to(self.device)
        text_len = len(text)
        return [feat, feat_len, text, text_len]

    def merge_batch(self, main_batch, attach_batch):
        max_feat_len = max(main_batch[1])
        max_text_len = max(main_batch[3])
        if attach_batch[0].shape[1] > max_feat_len:
            # reduce extra long example
            attach_batch[0] = attach_batch[0][:,:max_feat_len]
            attach_batch[1][0] = max_feat_len
        else:
            # pad to max_feat_len
            padding = torch.zeros(1, max_feat_len - attach_batch[0].shape[1], attach_batch[0].shape[2], dtype=attach_batch[0].dtype).to(self.device)
            attach_batch[0] = torch.cat([attach_batch[0], padding], dim=1)
        if attach_batch[2].shape[0] > max_text_len:
            attach_batch[2] = attach_batch[2][:max_text_len]
            main_batch[3][0] = max_text_len
        else:
            padding = torch.zeros(max_text_len - attach_batch[2].shape[0], dtype=attach_batch[2].dtype).to(self.device)
            try:
                attach_batch[2] = torch.cat([attach_batch[2], padding], dim=0).unsqueeze(0)
            except:
                pdb.set_trace()
        new_batch = (
            torch.cat([main_batch[0], attach_batch[0]], dim=0),
            torch.cat([main_batch[1], attach_batch[1]], dim=0),
            torch.cat([main_batch[2], attach_batch[2]], dim=0),
            torch.cat([main_batch[3], torch.tensor([attach_batch[3]]).to(self.device)], dim=0)
        )
        return new_batch
            


    def finetune(self, filename, fixed_text, max_step=5):
        # Load data for finetune
        self.verbose('Total training steps {}.'.format(
            human_format(max_step)))
        ctc_loss, att_loss, emb_loss = None, None, None
        n_epochs = 0
        accum_count = 0
        self.timer.set()
        step = 0
        for data in self.tr_set:
            # Pre-step : update tf_rate/lr_rate and do zero_grad
            if max_step == 0:
                break
            tf_rate = self.optimizer.pre_step(400000)
            total_loss = 0

            # Fetch data
            finetune_data = self.fetch_finetune_data(filename, fixed_text)
            main_batch = self.fetch_data(data)
            new_batch = self.merge_batch(main_batch, finetune_data)
            feat, feat_len, txt, txt_len = new_batch
            self.timer.cnt('rd')

            # Forward model
            # Note: txt should NOT start w/ <sos>
            ctc_output, encode_len, att_output, att_align, dec_state = \
                self.model(feat, feat_len, max(txt_len), tf_rate=tf_rate,
                            teacher=txt, get_dec_state=self.emb_reg)

            # Plugins
            if self.emb_reg:
                emb_loss, fuse_output = self.emb_decoder(
                    dec_state, att_output, label=txt)
                total_loss += self.emb_decoder.weight*emb_loss

            # Compute all objectives
            if ctc_output is not None:
                if self.paras.cudnn_ctc:
                    ctc_loss = self.ctc_loss(ctc_output.transpose(0, 1),
                                                txt.to_sparse().values().to(device='cpu', dtype=torch.int32),
                                                [ctc_output.shape[1]] *
                                                len(ctc_output),
                                                txt_len.cpu().tolist())
                else:
                    ctc_loss = self.ctc_loss(ctc_output.transpose(
                        0, 1), txt, encode_len, txt_len)
                total_loss += ctc_loss*self.model.ctc_weight

            if att_output is not None:
                b, t, _ = att_output.shape
                att_output = fuse_output if self.emb_fuse else att_output
                att_loss = self.seq_loss(
                    att_output.contiguous().view(b*t, -1), txt.contiguous().view(-1))
                total_loss += att_loss*(1-self.model.ctc_weight)

            self.timer.cnt('fw')

            # Backprop
            grad_norm = self.backward(total_loss)
            step += 1

            # Logger
            self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                        .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
            self.write_log(
                'loss', {'tr_ctc': ctc_loss, 'tr_att': att_loss})
            self.write_log('emb_loss', {'tr': emb_loss})
            self.write_log('wer', {'tr_att': cal_er(self.tokenizer, att_output, txt),
                                'tr_ctc': cal_er(self.tokenizer, ctc_output, txt, ctc=True)})
            if self.emb_fuse:
                if self.emb_decoder.fuse_learnable:
                    self.write_log('fuse_lambda', {
                                'emb': self.emb_decoder.get_weight()})
                self.write_log(
                    'fuse_temp', {'temp': self.emb_decoder.get_temp()})

            # End of step
            # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
            torch.cuda.empty_cache()
            self.timer.set()
            if step > max_step:
                break
        ret = self.validate()
        self.log.close()
        return ret


    def validate(self):
        # Eval mode
        self.model.eval()
        if self.emb_decoder is not None:
            self.emb_decoder.eval()
        dev_wer = {'att': [], 'ctc': []}

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model(feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO),
                               emb_decoder=self.emb_decoder)

            dev_wer['att'].append(cal_er(self.tokenizer, att_output, txt))
            dev_wer['ctc'].append(cal_er(self.tokenizer, ctc_output, txt, ctc=True))

            # Show some example on tensorboard
            if i == len(self.dv_set)//2:
                for i in range(min(len(txt), self.DEV_N_EXAMPLE)):
                    if True:
                        self.write_log('true_text{}'.format(
                            i), self.tokenizer.decode(txt[i].tolist()))
                    if att_output is not None:
                        self.write_log('att_align{}'.format(i), feat_to_fig(
                            att_align[i, 0, :, :].cpu().detach()))
                        self.write_log('att_text{}'.format(i), self.tokenizer.decode(
                            att_output[i].argmax(dim=-1).tolist()))
                    if ctc_output is not None:
                        self.write_log('ctc_text{}'.format(i), self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
                                                                                     ignore_repeat=True))

        # Skip save model here
        # Ckpt if performance improves
        to_prints = []
        for task in ['att', 'ctc']:
            dev_wer[task] = sum(dev_wer[task]) / len(dev_wer[task])
            if dev_wer[task] < self.best_wer[task]:
                to_print = f"WER of {task}: {dev_wer[task]} < prev best ({self.best_wer[task]})"
                self.best_wer[task] = dev_wer[task]
            else:
                to_print = f"WER of {task}: {dev_wer[task]} >= prev best ({self.best_wer[task]})"
            print(to_print, flush=True)
            to_prints.append(to_print)
        #         self.save_checkpoint('best_{}.pth'.format(task), 'wer', dev_wer[task])
            self.write_log('wer', {'dv_'+task: dev_wer[task]})
        # self.save_checkpoint('latest.pth', 'wer', dev_wer['att'], show_msg=False)

        # Resume training
        self.model.train()
        if self.emb_decoder is not None:
            self.emb_decoder.train()
        return '\n'.join(to_prints)



