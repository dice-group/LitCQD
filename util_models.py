import os
import torch
from models import *
from util_data import get_all_attribute_values, get_all_entity_descriptions, load_mappings_from_file
from util import query_name_dict
from config import CQDParams, HyperParams, TrainConfig


def get_model(train_config: TrainConfig, params: HyperParams, cqd_params: CQDParams, nentity, nrelation, nattribute):
    if type(train_config.geo) != str:
        geo = train_config.geo.name
    else:
        geo = train_config.geo
    if geo == 'cqd-transea':
        model = CQDTransEA(
            p_norm=params.p_norm,
            use_attributes=train_config.use_attributes,
            do_sigmoid=params.do_sigmoid,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-transeadistmult':
        model = CQDTransEADistMult(
            use_attributes=train_config.use_attributes,
            do_sigmoid=params.do_sigmoid,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-transeacomplex':
        model = CQDTransEAComplEx(
            use_modulus=params.use_modulus,
            use_attributes=train_config.use_attributes,
            do_sigmoid=params.do_sigmoid,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-transra':
        model = CQDTransRA(
            p_norm=params.p_norm,
            use_attributes=train_config.use_attributes,
            rank_attr=params.rank_attr,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-mtkgnn':
        model = MTKGNN(nentity, nrelation, nattribute, params.rank, params.p_norm)
    elif geo == 'cqd-transcomplexa':
        model = CQDTransComplExA(
            do_sigmoid=params.do_sigmoid,
            p_norm=params.p_norm,
            use_attributes=train_config.use_attributes,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-complex':
        model = CQDComplEx(
            do_sigmoid=False,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-complexa':
        model = CQDComplExA(
            use_attributes=train_config.use_attributes,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-complexa-weighted':
        model = CQDComplExAWeighted(
            attr_values=get_all_attribute_values(train_config.data_path),
            use_attributes=train_config.use_attributes,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-complexad':
        model = CQDComplExAD(
            word_emb_dim=train_config.word_emb_dim,
            desc_emb=params.desc_emb.name if type(params.desc_emb) != str else params.desc_emb,
            use_attributes=train_config.use_attributes,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-complexd':
        if type(params.desc_emb) != str:
            params.desc_emb = params.desc_emb.name
        model = CQDComplExD(
            word_emb_dim=train_config.word_emb_dim,
            desc_emb=params.desc_emb,
            use_attributes=train_config.use_attributes,
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name if type(cqd_params.cqd_type) != str else cqd_params.cqd_type,
            t_norm_name=cqd_params.cqd_t_norm.name if type(cqd_params.cqd_t_norm) != str else cqd_params.cqd_t_norm,
            k=cqd_params.cqd_k,
        )
    elif geo == 'cqd-complexd-jointly':
        model = CQDComplExDJointly(
            word_emb_dim=train_config.word_emb_dim,
            descriptions=get_all_entity_descriptions(train_config.data_path) if train_config.do_train else dict(),
            word2id=load_mappings_from_file(train_config.data_path, 'word'),
            rank=params.rank,
            nentity=nentity,
            nrelation=nrelation,
            nattr=nattribute,
            query_name_dict=query_name_dict,
            method=cqd_params.cqd_type.name,
            t_norm_name=cqd_params.cqd_t_norm.name,
            k=cqd_params.cqd_k,
        )
    elif geo == 'q2b':
        model = Query2Box(nentity, nrelation, nattribute, train_config.use_attributes, params.rank, use_cuda=train_config.cuda)
    elif geo == 'gqe':
        model = GQE(nentity, nrelation, nattribute, params.rank, params.margin, use_cuda=train_config.cuda)
    elif geo == 'random_guesser':
        model = RandomGuesser(nentity)

    if train_config.cuda:
        model = model.cuda()
    return model


def load_model(model, save_path, cuda, remove_attribute_exists=False):
    checkpoint = os.path.join(save_path, 'checkpoint')
    
    data = torch.load(checkpoint, map_location=torch.device('cuda:0') if cuda else torch.device('cpu'))

    # checkpoint may include optimizer state
    if type(data) != tuple:
        model_state = data['model_state_dict']
    else:
        model_state, _ = data

    if model_state['ent_embeddings.weight'].shape[0] == model.ent_embeddings.weight.shape[0] + 1:
        # Remove dummy entity and relations from checkpoint
        remove_attribute_exists = True

    if remove_attribute_exists:
        model_state['ent_embeddings.weight'] = model_state['ent_embeddings.weight'][:-1]
        model_state['rel_embeddings.weight'] = model_state['rel_embeddings.weight'][:474]
        if 'description_embeddings.weight' in model_state:
            model_state['description_embeddings.weight'] = model_state['description_embeddings.weight'][:-1]
        if 'offset_embedding.weight' in model_state:
            model_state['offset_embedding.weight'] = model_state['offset_embedding.weight'][:474]

    # Remove attribute embeddings from model_state if model is evaluated without attributes
    for attr_emb in ('attr_embeddings', 'b', 'offset_attr_embeddings'):
        if attr_emb+'.weight' in model_state and not hasattr(model, attr_emb):
            del model_state[attr_emb+'.weight']

    model.load_state_dict(model_state)


def save_model(model, optimizer, save_variable_list, save_path):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint')
    )
