import os
from ray import tune
from tensorboardX.writer import SummaryWriter
import torch
import collections
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from torch.autograd import Variable

from Loss import Loss
from models import CQDBaseModel,CQDComplExAD
from dataloader import TrainDataset


class Trainer(object):
    def __init__(self,
                 model: CQDBaseModel,
                 dataloader: DataLoader,
                 data_loader_type,
                 use_gpu,
                 learning_rate,
                 learning_rate_attr,
                 save_path,
                 train_times,
                 optimizerClass,
                 rel_loss: Loss,
                 attr_loss: Loss,
                 alpha,
                 beta,
                 dataloader_attr=None,
                 dataloader_desc=None,
                 neg_ent_attr=0,
                 reg_weight_ent=0.0,
                 reg_weight_rel=0.0,
                 reg_weight_attr=0.0,
                 scheduler_patience=10,
                 scheduler_factor=0.1,
                 scheduler_threshold=1e-4
                 ):
        self.model = model
        self.dataloader = dataloader
        assert not (data_loader_type == 'cpp' and dataloader_attr != None)  # cpp dataloader loads attributes by itself
        self.dataloader_attr = dataloader_attr
        self.dataloader_desc = dataloader_desc

        self.use_gpu = use_gpu
        self.optimizer = optimizerClass(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            # weight_decay=1.55e-10 l2 regularization with same weight for ent/rel
        )
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=scheduler_patience, factor=scheduler_factor, threshold=scheduler_threshold)
        self.save_path = save_path
        self.train_times = train_times
        self.rel_loss = rel_loss
        self.attr_loss = attr_loss
        self.alpha = alpha
        self.beta = beta
        self.neg_ent_attr = neg_ent_attr
        self.reg_weight_ent = reg_weight_ent
        self.reg_weight_rel = reg_weight_rel
        self.reg_weight_attr = reg_weight_attr
        
        assert self.alpha >=0 and self.alpha<=1
        assert self.beta >=0 and self.beta<=1

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def _custom_l2_regularization(self):
        pen = None
        if self.reg_weight_ent > 0:
            ent_reg = self.reg_weight_ent * self.model.ent_embeddings.weight.norm(p=2) ** 2
            pen = ent_reg
        if self.reg_weight_rel > 0:
            rel_reg = self.reg_weight_rel * self.model.rel_embeddings.weight.norm(p=2) ** 2
            if pen:
                pen += rel_reg
            else:
                pen = rel_reg
        if hasattr(self.model, "attr_embeddings") and self.reg_weight_attr > 0:
            attr_reg = self.reg_weight_attr * self.model.attr_embeddings.weight.norm(p=2) ** 2
            if hasattr(self.model, "offset_attr_embeddings"):
                attr_reg += self.reg_weight_attr * self.model.offset_attr_embeddings.weight.norm(p=2) ** 2
            if pen:
                pen += attr_reg
            else:
                pen = attr_reg
        if pen:
            pen.backward()

    def _finish_step(self, rel_scores, attr_scores, subsampling_weight, do_eval=False):
        rel_loss = self.rel_loss.compute(rel_scores, subsampling_weight)
        if self.attr_loss is not None:
            attr_loss = self.attr_loss.compute(attr_scores).nan_to_num()  # if no attr data is used loss is nan
        else:
            attr_loss = 0
        loss = self.alpha * attr_loss + (1-self.alpha) * rel_loss
        if not do_eval:
            loss.backward()
            self._custom_l2_regularization()
            self.optimizer.step()
        return loss.item(), rel_loss.item(), attr_loss.item() if attr_loss != 0 else attr_loss

    def train(self, write_loss_fn, eval_fn=None, eval_epochs=100):
        # get rid of eval_fn
        training_range = tqdm(range(self.train_times)) # initialize progress bar
        
        
        
        for epoch in training_range:
            
            # if eval_fn and epoch % eval_epochs == 0 and epoch > 0 or epoch == 1:
            #     eval_fn(epoch)

            loss_per_step = list()
            dataloader_attr = self.dataloader_attr if self.dataloader_attr else [None]*len(self.dataloader)
            dataloader_desc = self.dataloader_desc if self.dataloader_desc else [None]*len(self.dataloader)
            for data, data_attr, data_desc in zip(self.dataloader, dataloader_attr, dataloader_desc):
                loss_per_step.append(self.train_step_python(data, data_attr=data_attr, data_desc=data_desc))

            loss_mean = sum([_[0] for _ in loss_per_step]) / len(loss_per_step)
            rel_loss_mean = sum([_[1] for _ in loss_per_step]) / len(loss_per_step)
            attr_loss_mean = sum([_[2] for _ in loss_per_step]) / len(loss_per_step)

            training_range.set_description("Epoch %d | relational/attribute loss: %.3f/%.3f | lr: %f" % (epoch, rel_loss_mean, attr_loss_mean, self.optimizer.param_groups[0]['lr']))
            
            if write_loss_fn:
                write_loss_fn(loss_mean, rel_loss_mean, attr_loss_mean, epoch + 1)

            # self.scheduler.step(loss_mean)


    # def train(self, eval_fn, write_loss_fn, eval_epochs=100):
    #     # get rid of eval_fn
    #     training_range = tqdm(range(self.train_times))
    #     for epoch in training_range:
    #         if epoch % eval_epochs == 0 and epoch > 0 or epoch == 1:
    #             eval_fn(epoch)

    #         loss_per_step = list()
    #         dataloader_attr = self.dataloader_attr if self.dataloader_attr else [None]*len(self.dataloader)
    #         dataloader_desc = self.dataloader_desc if self.dataloader_desc else [None]*len(self.dataloader)
    #         for data, data_attr, data_desc in zip(self.dataloader, dataloader_attr, dataloader_desc):
    #             loss_per_step.append(self.train_step_python(data, data_attr, data_desc))

    #         loss_mean = sum([_[0] for _ in loss_per_step]) / len(loss_per_step)
    #         rel_loss_mean = sum([_[1] for _ in loss_per_step]) / len(loss_per_step)
    #         attr_loss_mean = sum([_[2] for _ in loss_per_step]) / len(loss_per_step)

    #         training_range.set_description("Epoch %d | relational/attribute loss: %.3f/%.3f | lr: %f" % (epoch, rel_loss_mean, attr_loss_mean, self.optimizer.param_groups[0]['lr']))
    #         write_loss_fn(loss_mean, rel_loss_mean, attr_loss_mean, epoch + 1)

    #         # self.scheduler.step(loss_mean)

    def train_ray_desc(self, eval_fn, eval_epochs=20):
        training_range = range(self.train_times)
        for epoch in training_range:

            loss_per_step = list()
            dataloader_attr = self.dataloader_attr if self.dataloader_attr else [None]*len(self.dataloader)
            dataloader_desc = self.dataloader_desc if self.dataloader_desc else [None]*len(self.dataloader)
            for data, data_attr, data_desc in zip(self.dataloader, dataloader_attr, dataloader_desc):
                loss_per_step.append(self.train_step_python(data, data_attr, data_desc))

            loss_mean = sum([_[0] for _ in loss_per_step]) / len(loss_per_step)

            # self.scheduler.step(loss_mean)
            if (epoch + 1) % eval_epochs == 0 or epoch == 1:
                metrics = eval_fn(epoch+1)
                rel_metric = metrics['1p_MRR']
                desc_metric = metrics['1dp_cos_sim']
                desc_metric2 = metrics['di_MRR']
                desc_metric_train = metrics['train_1dp_cos_sim']

                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

                tune.report(mrr=rel_metric, cos_sim=desc_metric, train_cos_sim=desc_metric_train, di_mrr=desc_metric2, loss=loss_mean)

    def train_ray(self, eval_fn, valid_dataloader, valid_attr_dataloader, eval_epochs=20):
        training_range = range(self.train_times)
        for epoch in training_range:

            loss_per_step = list()
            for data, data_attr in zip(self.dataloader, self.dataloader_attr) if self.dataloader_attr else zip(self.dataloader, [None]*len(self.dataloader)):
                loss_per_step.append(self.train_step_python(data, data_attr))

            #loss_mean = sum([_[0] for _ in loss_per_step]) / len(loss_per_step)

            # self.scheduler.step(loss_mean)
            if (epoch + 1) % eval_epochs == 0:
                valid_loss_per_step = list()
                for data, data_attr in zip(valid_dataloader, valid_attr_dataloader) if valid_attr_dataloader else zip(valid_dataloader, [None]*len(valid_dataloader)):
                    valid_loss_per_step.append(self.train_step_python(data, data_attr, do_eval=True))

                valid_loss_mean = sum([_[0] for _ in valid_loss_per_step]) / len(valid_loss_per_step)
                valid_rel_loss_mean = sum([_[1] for _ in valid_loss_per_step]) / len(valid_loss_per_step)
                valid_attr_loss_mean = sum([_[2] for _ in valid_loss_per_step]) / len(valid_loss_per_step)

                metrics = eval_fn(epoch+1)
                rel_metric = metrics['1p_MRR']
                if '1ap_MAE' in metrics:
                    attr_metric = metrics['1ap_MAE']
                    attr_metric2 = metrics['1ap_MSE']
                    score = rel_metric - attr_metric
                else:
                    attr_metric, attr_metric2 = 0, 0
                    score = rel_metric

                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

                tune.report(score=score, mrr=rel_metric, mae=attr_metric, mse=attr_metric2, loss=valid_loss_mean, rel_loss=valid_rel_loss_mean, attribute_loss=valid_attr_loss_mean)

    # TODO: train_step()
    def train_step_python(self, data, data_attr=None, data_desc=None, do_eval=False):
        # data_desc, do_eval should not be used
        if do_eval:
            # Used to get a loss on validation set for ray tune
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        device = torch.device("cuda:0" if self.use_gpu else "cpu")

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = data
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        # group queries with same structure
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).to(device=device)
        if self.use_gpu:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # Reorder samples for query dataloader compatibility
        all_idxs = []
        for qs in batch_queries_dict:
            all_idxs.extend(batch_idxs_dict[qs])
        positive_sample = positive_sample[all_idxs]
        negative_sample = negative_sample[all_idxs]

        if type(self.dataloader.dataset) is TrainDataset:
            # train using queries, which may be complex (queries other than 1p, 1ap)
            samples = torch.cat((positive_sample.unsqueeze(-1), negative_sample), dim=-1)
            scores = self.model(batch_queries_dict, samples)  # forward()
            scores = torch.stack(scores, dim=0)
            rel_scores = torch.cat((scores[..., 0], scores[..., 1:].reshape((-1,))), dim=0)
            if data_attr:
                data = {
                    "batch_e": torch.flatten(data_attr[0]).to(device=device),
                    "batch_a": torch.flatten(data_attr[1]).to(device=device),
                    "batch_v": torch.flatten(data_attr[2]).to(device=device),
                    "attr_pos_samples": data_attr[0].shape[0],
                }
                attr_scores = self.model.score_attr(data) # get the scores of attributes
            else:
                attr_scores = torch.FloatTensor().to(device=device) # empty tensor
        else:
            # train using triples (CQD Dataloader)
            input_batch = batch_queries_dict[('e', ('r',))]

            data = {
                "batch_h": input_batch[:, 0].repeat(1+negative_sample.shape[-1]).to(device=device),
                "batch_r": input_batch[:, 1].repeat(1+negative_sample.shape[-1]).to(device=device),
                "batch_t": torch.cat((positive_sample, torch.flatten(negative_sample))).to(device=device),
                "mode": "normal",
            }
            if data_attr:
                data["batch_e"] = torch.flatten(data_attr[0]).to(device=device)
                data["batch_a"] = torch.flatten(data_attr[1]).to(device=device)
                data["batch_v"] = torch.flatten(data_attr[2]).to(device=device)
                data["attr_pos_samples"] = data_attr[0].shape[0]

            if data_desc:
                data['batch_desc_e'] = torch.flatten(data_desc[0]).to(device=device)
                data['batch_desc_d'] = data_desc[1].to(device=device)

            legacy_mode = hasattr(self.model, 'loss') and callable(self.model.loss)
            if legacy_mode:
                # Loss function part of the model (e.g. ComplEx-N3)

                def attr_loss_fn(scores): return self.attr_loss.compute(scores).nan_to_num() if self.attr_loss is not None else 0
                
                if isinstance(self.model,CQDComplExAD):
                  loss = self.model.loss(data, attr_loss_fn, self.alpha,self.beta)
                else:
                  loss = self.model.loss(data, attr_loss_fn, self.alpha)
                  
                # loss = self.model.loss(data, attr_loss_fn)
                if not do_eval:
                    loss.backward()
                    self.optimizer.step()
                return loss.item(), loss.item(), 0
            
            rel_scores, attr_scores = self.model.score(data)
        
        return self._finish_step(rel_scores, attr_scores, subsampling_weight, do_eval)
