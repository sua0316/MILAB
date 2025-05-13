import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
import torch.nn.functional as F
from ..subNets import BertTextEncoder
import random

logger = logging.getLogger('MMSA')

def get_sentiment_class(label):
    if label < -2.5:
        return 0
    elif label >=-2.5 and label < -1.5:
        return 1
    elif label >=-1.5 and label < -0.5:
        return 2
    elif label >= -0.5 and label < 0.5:
        return 3
    elif label >= 0.5 and label < 1.5:
        return 4
    elif label >= 1.5 and label < 2.5:
        return 5
    elif label >= 2.5:
        return 6

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DMD():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        #if args.use_bert:
        #   self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
        #                                      pretrained=args.pretrained)
        #self.use_bert = args.use_bert
        #self.text_model = self.text_model.cuda()

    def do_train(self, model, dataloader, return_epoch_results=False):
        # 0: DMD model, 1: Homo GD, 2: Hetero GD
        params = list(model[0].parameters()) + \
                 list(model[1].parameters()) + \
                 list(model[2].parameters())

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_dmd = model[0]
        net_distill_homo = model[1]
        net_distill_hetero = model[2]
        net.append(net_dmd)
        net.append(net_distill_homo)
        net.append(net_distill_hetero)
        model = net

        while True:
            epochs += 1
            
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    '''''
                    
                    if self.use_bert:
                        text = self.text_model(text)
                    
                    text_features = torch.reshape(text, (text.shape[0], 50*768)) # [16, 38400]
                    #text_features = torch.flatten(text, start_dim=1)  # (batch_size, -1) 배치 크기 유지, 나머지 평탄화
                    audio_features = torch.reshape(audio, (audio.shape[0], 50*5))
                    video_features = torch.reshape(vision, (vision.shape[0], 50*20))
                    
                    #similarity
                    text_similarity = F.cosine_similarity(text_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)
                    audio_similarity = F.cosine_similarity(audio_features.unsqueeze(1), audio_features.unsqueeze(0), dim=-1)
                    video_similarity = F.cosine_similarity(video_features.unsqueeze(1), video_features.unsqueeze(0), dim=-1)

                    #combine
                    combined_similarity = (text_similarity + audio_similarity + video_similarity) / 3
                    
                    batch_size = combined_similarity.size(0)
                    
                    for i in range(batch_size):
                        for j in range(batch_size):
                            if get_sentiment_class(labels[i])  != get_sentiment_class(labels[j]):
                                combined_similarity[i, j] = -float('inf')
                    
                    
                    for i in range(batch_size):
                        combined_similarity[i, i] = -float('inf')
                    most_similar_indices = combined_similarity.topk(k=1, dim=-1).indices.squeeze(-1)

                    
                    for i in range(batch_size):
                        flag = 0
                        for j in range(batch_size):
                            if combined_similarity[i,j] != -float('inf'):
                                flag = 1
                        if flag == 0:
                            most_similar_indices[i] = i
                    
                    '''''
                    
                    original_output = model[0](text, audio, vision, is_distill=True)
                    origin_l, origin_v, origin_a, s_l, s_v, s_a, c_l, c_v, c_a, last_hs, last_hs_proj, proj_s_l, proj_s_v, proj_s_a, c_l_att, c_v_att, c_a_att =\
                         original_output['origin_l'], original_output['origin_v'], original_output['origin_a'], original_output['s_l'], original_output['s_v'], original_output['s_a'], \
                         original_output['c_l'], original_output['c_v'], original_output['c_a'], original_output['last_hs'], original_output['last_hs_proj'], \
                         original_output['proj_s_l'], original_output['proj_s_v'], original_output['proj_s_a'], original_output['c_l_att'], original_output['c_v_att'], original_output['c_a_att']

                    '''''
                    low_level_features = torch.cat([
                        origin_l.reshape(origin_l.shape[0], -1),  # [Batch, Flattened]
                        origin_a.reshape(origin_a.shape[0], -1),  # [Batch, Flattened]
                        origin_v.reshape(origin_v.shape[0], -1)   # [Batch, Flattened]
                    ], dim=-1)  # [Batch, Combined Feature]
                    low_level_similarity = F.cosine_similarity(
                        low_level_features.unsqueeze(1), 
                        low_level_features.unsqueeze(0), 
                        dim=-1
                    )                    

                    specific_features = torch.cat([proj_s_l, proj_s_v, proj_s_a], dim=-1)
                    #print(proj_s_l.shape, proj_s_v.shape, proj_s_a.shape, specific_features.shape)
                    #exit(0)
                    #specific_features = specific_features.permute(1, 0, 2)  # [46, 16, 150] -> [16, 46, 150]
                    #specific_features = specific_features.reshape(specific_features.shape[0], -1)  # [16, 6900]

                    specific_similarity = F.cosine_similarity(
                        specific_features.unsqueeze(1), 
                        specific_features.unsqueeze(0), 
                        dim=-1
                    )

                    invariant_features = torch.cat([c_l_att, c_v_att, c_a_att], dim=-1)
                    #print(c_l_att.shape, c_v_att.shape, c_a_att.shape, invariant_features.shape)
                    #exit(0)
                    #invariant_features = invariant_features.permute(1, 0, 2)  # [46, 16, 150] -> [16, 46, 150]
                    #invariant_features = invariant_features.reshape(invariant_features.shape[0], -1)  # [16, 6900]

                    invariant_similarity = F.cosine_similarity(
                        invariant_features.unsqueeze(1), 
                        invariant_features.unsqueeze(0), 
                        dim=-1
                    )
                    '''''
                    '''''
                    # Low-level Feature MSE
                    low_level_features = torch.cat([
                        origin_l.reshape(origin_l.shape[0], -1),
                        origin_a.reshape(origin_a.shape[0], -1),
                        origin_v.reshape(origin_v.shape[0], -1)
                    ], dim=-1)

                    low_level_similarity = -((low_level_features.unsqueeze(1) - low_level_features.unsqueeze(0)) ** 2).mean(dim=-1)

                    # Specific Features MSE
                    specific_features = torch.cat([proj_s_l, proj_s_v, proj_s_a], dim=-1)
                    #specific_features = specific_features.permute(1, 0, 2)
                    #specific_features = specific_features.reshape(specific_features.shape[0], -1)

                    specific_similarity = -((specific_features.unsqueeze(1) - specific_features.unsqueeze(0)) ** 2).mean(dim=-1)

                    # Invariant Features MSE
                    invariant_features = torch.cat([c_l_att, c_v_att, c_a_att], dim=-1)
                    #invariant_features = invariant_features.permute(1, 0, 2)
                    #invariant_features = invariant_features.reshape(invariant_features.shape[0], -1)

                    invariant_similarity = -((invariant_features.unsqueeze(1) - invariant_features.unsqueeze(0)) ** 2).mean(dim=-1)
                    '''''
                    # High-level Features MSE
                    last_hs_proj_similarity = -((last_hs_proj.unsqueeze(1) - last_hs_proj.unsqueeze(0)) ** 2).mean(dim=-1)
                    
                    #'''''
                    # 기존 combined_features 대신 last_hs 사용
                    #similarity_matrix = F.cosine_similarity(last_hs.unsqueeze(1), last_hs.unsqueeze(0), dim=-1)

                    # 기존 combined_features 대신 last_hs_proj 사용
                    #last_hs_proj_similarity = F.cosine_similarity(last_hs_proj.unsqueeze(1), last_hs_proj.unsqueeze(0), dim=-1)
                    similarity_matrix = last_hs_proj_similarity

                    '''''
                    similarity_matrix = (
                        0.25 * low_level_similarity +
                        0.25 * specific_similarity +
                        0.25 * invariant_similarity +
                        0.25 * last_hs_proj_similarity
                    )
                    '''''
                    batch_size = similarity_matrix.size(0)
                    
                    for i in range(batch_size):
                        for j in range(batch_size):
                            if get_sentiment_class(labels[i])  != get_sentiment_class(labels[j]):
                                similarity_matrix[i, j] = -float('inf')
                    
                    for i in range(batch_size):
                        similarity_matrix[i, i] = -float('inf')
                    
                    most_similar_indices = similarity_matrix.topk(k=1, dim=-1).indices.squeeze(-1)
                    #most_similar_indices = similarity_matrix.topk(k=2, dim=-1).indices.squeeze(-1)
                    
                    for i in range(batch_size):
                        flag = 0
                        for j in range(batch_size):
                            if similarity_matrix[i,j] != -float('inf'):
                                flag = 1
                        if flag == 0:
                            most_similar_indices[i] = i
                    #'''''

                    #fused_labels = (labels + labels[most_similar_indices]) / 2
                    
                    augmented_text = (text + text[most_similar_indices]) / 2
                    augmented_audio = (audio + audio[most_similar_indices]) / 2
                    augmented_video = (vision + vision[most_similar_indices]) / 2

                    # 2. extrapolation
                    # extrapolation: Xˆ = (Xi − Xj ) ∗ λ + Xi
                    #lamda = 0.5
                    #augmented_text = (text - text[most_similar_indices]) * 0.5 + text
                    #augmented_audio = (audio - audio[most_similar_indices]) * 0.5 + audio
                    #augmented_video = (vision - vision[most_similar_indices]) * 0.5 + vision

                    # 3. linear delta
                    # linear delta: Xˆ = (Xi − Xj ) + Xk
                    #augmented_text = (text[most_similar_indices[:,0]] - text[most_similar_indices[:,1]]) + text
                    #augmented_audio = (audio[most_similar_indices[:,0]] -  audio[most_similar_indices[:,1]]) + audio
                    #augmented_video = (vision[most_similar_indices[:,0]] - vision[most_similar_indices[:,1]]) + vision

                    '''''
                    # cosine_similarity
                    text_similarities = F.cosine_similarity(text, text[most_similar_indices])  # Calculate similarity
                    text_similarities = text_similarities.clamp(min=0, max=1)
                    text_w_similar = text_similarities / (text_similarities + 1e-8)
                    text_w_original = 1 - text_w_similar

                    augmented_text = text_w_original[:, None] * text + text_w_similar[:, None] * text[most_similar_indices]
                    
                    audio_similarities = F.cosine_similarity(audio, audio[most_similar_indices])  # Calculate similarity
                    audio_similarities = audio_similarities.clamp(min=0, max=1)
                    audio_w_similar = audio_similarities / (audio_similarities + 1e-8)
                    audio_w_original = 1 - audio_w_similar

                    augmented_audio = audio_w_original[:, None] * audio + audio_w_similar[:, None] * audio[most_similar_indices]

                    video_similarities = F.cosine_similarity(vision, vision[most_similar_indices])  # Calculate similarity
                    video_similarities = video_similarities.clamp(min=0, max=1)
                    video_w_similar = video_similarities / (video_similarities + 1e-8)
                    video_w_original = 1 - video_w_similar

                    augmented_video = video_w_original[:, None] * vision + video_w_similar[:, None] * vision[most_similar_indices]
                    '''''
                    '''''
                    # Euclidean Distance
                    text_distances = torch.norm(text - text[most_similar_indices], dim=1)  # Euclidean distance
                    text_distances = text_distances.clamp(min=0, max=text_distances.max())

                    text_similarities = 1 / (1 + text_distances)

                    text_w_similar = text_similarities / (text_similarities + 1e-8)
                    text_w_original = 1 - text_w_similar

                    augmented_text = text_w_original[:, None] * text + text_w_similar[:, None] * text[most_similar_indices]

                    audio_distances = torch.norm(audio - audio[most_similar_indices], dim=1)  # Euclidean distance
                    audio_distances = audio_distances.clamp(min=0, max=audio_distances.max())
                    audio_similarities = 1 / (1 + audio_distances)
                    audio_w_similar = audio_similarities / (audio_similarities + 1e-8)
                    audio_w_original = 1 - audio_w_similar

                    augmented_audio = audio_w_original[:, None] * audio + audio_w_similar[:, None] * audio[most_similar_indices]

                    video_distances = torch.norm(vision - vision[most_similar_indices], dim=1)  # Euclidean distance
                    video_distances = video_distances.clamp(min=0, max=video_distances.max())
                    video_similarities = 1 / (1 + video_distances)
                    video_w_similar = video_similarities / (video_similarities + 1e-8)
                    video_w_original = 1 - video_w_similar

                    augmented_video = video_w_original[:, None] * vision + video_w_similar[:, None] * vision[most_similar_indices]
                    '''''
                    '''''
                    # Additive noise parameters
                    noise_std = 0.15

                    text_noise = torch.randn_like(text) * noise_std  # Random noise
                    augmented_text = (text + text[most_similar_indices]) / 2 + text_noise

                    audio_noise = torch.randn_like(audio) * noise_std
                    augmented_audio = (audio + audio[most_similar_indices]) / 2 + audio_noise

                    video_noise = torch.randn_like(vision) * noise_std
                    augmented_video = (vision + vision[most_similar_indices]) / 2 + video_noise
                    '''''

                    logits_homo, reprs_homo, logits_hetero, reprs_hetero = [], [], [], []

                    #original_output = model[0](text, audio, vision, is_distill=True)
                    augmented_output = model[0](augmented_text, augmented_audio, augmented_video, is_distill=True)

                    # logits for homo GD
                    logits_homo.append(augmented_output['logits_l_homo'])
                    logits_homo.append(augmented_output['logits_v_homo'])
                    logits_homo.append(augmented_output['logits_a_homo'])

                    # reprs for homo GD
                    reprs_homo.append(augmented_output['repr_l_homo'])
                    reprs_homo.append(augmented_output['repr_v_homo'])
                    reprs_homo.append(augmented_output['repr_a_homo'])

                    # logits for hetero GD
                    logits_hetero.append(augmented_output['logits_l_hetero'])
                    logits_hetero.append(augmented_output['logits_v_hetero'])
                    logits_hetero.append(augmented_output['logits_a_hetero'])

                    # reprs for hetero GD
                    reprs_hetero.append(augmented_output['repr_l_hetero'])
                    reprs_hetero.append(augmented_output['repr_v_hetero'])
                    reprs_hetero.append(augmented_output['repr_a_hetero'])

                    logits_homo = torch.stack(logits_homo)
                    reprs_homo = torch.stack(reprs_homo)

                    logits_hetero = torch.stack(logits_hetero)
                    reprs_hetero = torch.stack(reprs_hetero)

                    # edges for homo distill
                    edges_homo, edges_origin_homo = model[1](logits_homo, reprs_homo)

                    # edges for hetero distill
                    edges_hetero, edges_origin_hetero = model[2](logits_hetero, reprs_hetero)

                    
                    # task loss for original data
                    loss_task_all = self.criterion(original_output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(original_output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(original_output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(original_output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(original_output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(original_output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(original_output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(original_output['logits_c'], labels)
                    original_output_loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c
                    #loss_task = original_output_loss_task
                    
                    # task loss for augmented data
                    loss_task_all = self.criterion(augmented_output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(augmented_output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(augmented_output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(augmented_output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(augmented_output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(augmented_output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(augmented_output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(augmented_output['logits_c'], labels)
                    augmented_output_loss_task = loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c

                    # output distributions of original data and augmented data should be similar 
                    loss_similarity = self.criterion(original_output['output_logit'], augmented_output['output_logit'] ) + \
                                      self.criterion(original_output['logits_l_homo'], augmented_output['logits_l_homo'] ) + \
                                      self.criterion(original_output['logits_v_homo'], augmented_output['logits_v_homo'] ) + \
                                      self.criterion(original_output['logits_a_homo'], augmented_output['logits_a_homo'] ) + \
                                      self.criterion(original_output['logits_l_hetero'], augmented_output['logits_l_hetero'] ) + \
                                      self.criterion(original_output['logits_v_hetero'], augmented_output['logits_v_hetero'] ) + \
                                      self.criterion(original_output['logits_a_hetero'], augmented_output['logits_a_hetero'] ) + \
                                      self.criterion(original_output['logits_c'], augmented_output['logits_c'] )
                    loss_task = original_output_loss_task + augmented_output_loss_task + 0.1 * loss_similarity
                    

                    '''''
                    weight_original = 0.5
                    weight_augmented = 0.7
                    weight_similarity = 0.2

                    loss_task = (
                        weight_original * original_output_loss_task +
                        weight_augmented * augmented_output_loss_task +
                        weight_similarity * loss_similarity
                    )
                    '''''
                    '''''
                    total_loss = original_output_loss_task + augmented_output_loss_task + loss_similarity

                    weight_original = original_output_loss_task / total_loss
                    weight_augmented = augmented_output_loss_task / total_loss
                    weight_similarity = loss_similarity / total_loss

                    loss_task = (weight_original * original_output_loss_task +
                                weight_augmented * augmented_output_loss_task +
                                weight_similarity * loss_similarity)
                    '''''
                    '''
                    loss_task = original_output_loss_task  # Only original data's task loss
                    loss_similarity = self.criterion(original_output['output_logit'], augmented_output['output_logit'])
                    combined_loss = loss_task + 0.1 * loss_similarity

                    weight_task = 0.8
                    weight_similarity = 0.2
                    combined_loss = weight_task * loss_task + weight_similarity * loss_similarity
                    '''

                    # reconstruction loss
                    loss_recon_l = self.MSE(augmented_output['recon_l'], augmented_output['origin_l'])
                    loss_recon_v = self.MSE(augmented_output['recon_v'], augmented_output['origin_v'])
                    loss_recon_a = self.MSE(augmented_output['recon_a'], augmented_output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # cycle consistency loss between s_x and s_x_r
                    loss_sl_slr = self.MSE(augmented_output['s_l'].permute(1, 2, 0), augmented_output['s_l_r'])
                    loss_sv_slv = self.MSE(augmented_output['s_v'].permute(1, 2, 0), augmented_output['s_v_r'])
                    loss_sa_sla = self.MSE(augmented_output['s_a'].permute(1, 2, 0), augmented_output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss
                    cosine_similarity_s_c_l = self.cosine(augmented_output['s_l'].contiguous().view(labels.size(0),-1), augmented_output['c_l'].contiguous().view(labels.size(0),-1), torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_v = self.cosine(augmented_output['s_v'].contiguous().view(labels.size(0),-1), augmented_output['c_v'].contiguous().view(labels.size(0),-1), torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_a = self.cosine(augmented_output['s_a'].contiguous().view(labels.size(0),-1), augmented_output['c_a'].contiguous().view(labels.size(0),-1), torch.tensor([-1]).cuda()).mean(0)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # margin loss
                    c_l, c_v, c_a = augmented_output['c_l_sim'], augmented_output['c_v_sim'], augmented_output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    # homo GD loss
                    loss_reg_homo, loss_logit_homo, loss_repr_homo = \
                        model[1].distillation_loss(logits_homo, reprs_homo, edges_homo)
                    graph_distill_loss_homo = 0.05 * (loss_logit_homo + loss_reg_homo)

                    # hetero GD loss
                    loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
                        model[2].distillation_loss(logits_hetero, reprs_hetero, edges_hetero)
                    graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

                    combined_loss = loss_task + \
                                    graph_distill_loss_homo + graph_distill_loss_hetero + \
                                    (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1

                    combined_loss.backward()


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters()) + \
                                 list(model[1].parameters()) + \
                                 list(model[2].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()

                    y_pred.append(augmented_output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/dmd.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    #if self.use_bert:
                    #    text = self.text_model(text)
                    
                    output = model(text, audio, vision, is_distill=True)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results