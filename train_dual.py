import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AdamW, AutoModel, optimization, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

from models.ASFEN_model import ASFENClassifier
from data_utils import Tokenizer4BertGCN, ASFENData

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        self.opt = opt
        tokenizer1 = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)
        tokenizer = Tokenizer4BertGCN(tokenizer1, opt.max_length)
        bert = AutoModel.from_pretrained(opt.pretrained_bert_name)
        config = AutoConfig.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt, config).to(opt.device)
        trainset = ASFENData(opt.dataset_file['train'], tokenizer, opt=opt)
        valset = ASFENData(opt.dataset_file['val'], tokenizer, opt=opt)
        testset = ASFENData(opt.dataset_file['test'], tokenizer, opt=opt)

        # DistributedSampler
        train_sampler = DistributedSampler(trainset)
        val_sampler = DistributedSampler(valset)
        test_sampler = DistributedSampler(testset)

        # BatchSampler
        train_batch_sampler = BatchSampler(train_sampler, opt.batch_size, drop_last=True)

        # DataLoader
        self.train_dataloader = DataLoader(dataset=trainset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=2)
        self.val_dataloader = DataLoader(dataset=valset, batch_size=opt.batch_size, sampler=val_sampler, pin_memory=True, num_workers=2)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, sampler=test_sampler, pin_memory=True, num_workers=2)
        self.model = DistributedDataParallel(self.model, find_unused_parameters=True, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _train(self, criterion, optimizer, scheduler, max_test_acc_overall=0):
        max_val_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            losses = []
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                inputs_mask = sample_batched['attention_mask'].to(self.opt.device)
                outputs, penal = self.model(inputs, inputs_mask)
                targets = sample_batched['polarity'].to(self.opt.device)
                if self.opt.losstype is not None:
                    loss = criterion(outputs, targets) + penal
                else:
                    loss = criterion(outputs, targets)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    val_acc, f1 = self._evaluate(dataloader=self.val_dataloader)
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        if val_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, val_acc, f1)
                            self.best_model_path = model_path
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}, lr: {:.6f}'.format(loss.item(), train_acc, val_acc, f1, scheduler.get_last_lr()[0]))
                    # logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, val_acc, f1))

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

            logger.info('epoch: {}, loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(epoch, np.mean(losses), train_acc, max_val_acc, max_f1))

        return max_val_acc, max_f1, model_path

    def _evaluate(self, dataloader, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                inputs_mask = sample_batched['attention_mask'].to(self.opt.device)
                outputs, penal = self.model(inputs, inputs_mask)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1
        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        test_report, test_confusion, acc, f1 = self._evaluate(dataloader=self.test_dataloader, show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss().to(self.opt.device)
        optimizer = self.get_bert_optimizer(self.model)
        total_steps = len(self.train_dataloader) * self.opt.num_epoch
        scheduler = optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
        max_test_acc_overall = 0
        max_f1_overall = 0
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, scheduler, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()

def main():
    model_classes = {
        'ASFEN': ASFENClassifier,
    }

    dataset_files = {
        'twitter17': {
            'train': './dataset/Tweets17_corenlp/twitter17_train_write.json',
            'val': './dataset/Tweets17_corenlp/twitter17_val_write.json',
            'test': './dataset/Tweets17_corenlp/twitter17_test_write.json',
        },
        'twitter15': {
            'train': './dataset/Tweets15_corenlp/twitter15_train_write.json',
            'val': './dataset/Tweets15_corenlp/twitter15_val_write.json',
            'test': './dataset/Tweets15_corenlp/twitter15_test_write.json',
        }
    }

    input_colses = {
        'ASFEN': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                       'aspect_mask', 'short_mask', 's1_input_ids', 's1_attention_mask', 's2_input_ids', 's2_attention_mask']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,  # 权重初始化，将张量中的数值初始化为均匀分布中的随机值
        'xavier_normal_': torch.nn.init.xavier_normal_,  # 权重初始化，将张量中的数值初始化为给定均值和标准差的正态分布中的随机值
        'orthogonal_': torch.nn.init.orthogonal_,  # 权重初始化，将张量中的数值初始化为正交分布中的随机值
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
        'adamW': torch.optim.AdamW,
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ASFEN', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter17', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adamW', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=20, type=int)  # 10
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='GCN mem dim.')  # 768
    parser.add_argument('--num_layers', type=int, default=1, help='Num of GCN layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Tweets_corenlp')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--cuda', default='0,1', type=str)
    parser.add_argument('--losstype', default=None, type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)

    parser.add_argument('--vision_dropout', type=float, default=0.1, help='vision dropout rate.')
    parser.add_argument('--num_classes', type=int, default=3, help='num classes.')

    # * bert
    parser.add_argument('--pretrained_bert_name', default='/usr/data/xlq/huggingface/bert-large', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=1024)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=1e-5, type=float)

    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='rank of distributed processes')

    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    def init_distributed_mode(opt):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            opt.rank = int(os.environ["RANK"])
            opt.world_size = int(os.environ['WORLD_SIZE'])
            opt.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
            opt.distributed = False
            return

        opt.distributed = True
        opt.dist_url = 'env://'
        opt.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}'.format(opt.rank, opt.dist_url), flush=True)
        dist.init_process_group(backend=opt.dist_backend,
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.rank)
        dist.barrier()

    def cleanup():
        dist.destroy_process_group()

    # set random seed
    setup_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    init_distributed_mode(opt)
    device = torch.device(opt.device)
    torch.cuda.set_device(opt.local_rank)

    if not os.path.exists('./log'):
        os.makedirs('./log', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    ins = Instructor(opt)
    ins.run()
    cleanup()

if __name__ == '__main__':
    main()
