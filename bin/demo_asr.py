import copy
import torch
from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed

from src.solver import BaseSolver
from src.asr import ASR
from src.decode import BeamDecoder
from src.data import load_dataset, load_text_encoder
from src.online import Datadealer


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
        self.config['data']['text'] = self.src_config['data']['text']
        self.tokenizer = load_text_encoder(**self.config['data']['text'])
        self.config['model'] = self.src_config['model']

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

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.dv_set, self.tt_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.paras.njobs, self.paras.gpu,
                         self.paras.pin_memory, False, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model '''
        # Model
        self.feat_dim = 120
        self.vocab_size = 46 
        init_adadelta = True
        self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, 
                         **self.config['model'])

        # Plug-ins
        if ('emb' in self.config) and (self.config['emb']['enable']) \
                and (self.config['emb']['fuse'] > 0):
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb'])

        # Load target model in eval mode
        self.load_ckpt()

        # Beam decoder
        self.decoder = BeamDecoder(
            self.model.cpu(), self.emb_decoder, **self.config['decode'])
        self.verbose(self.decoder.create_msg())
        del self.model
        del self.emb_decoder
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



