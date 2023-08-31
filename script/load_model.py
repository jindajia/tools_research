import torch


def load_model(path):
    print(path)
    model = torch.load(path, map_location='cpu')
    print(type(model))
    print('type(model[0])=', type(model[0]))
    print(model[0].dtype, 'size = ', model[0].element_size(), 'Bytes')
    print(len(model))
    print('len =', model[0].shape[0])
    print('memory = {:.2f} '.format(model[0].element_size() * model[0].shape[0] / 1e6), 'MBytes\n')
    return model

def load_zero1_model(path):
    print(path)
    model = torch.load(path, map_location='cpu')
    print('type(model)=', type(model))
    layer = model['module']['encoder.encoder.layer.4.intermediate.dense.weight']
    print(layer.size())
    for id, key in enumerate(model['module']):
        print(key)
        print(model['module'][key].size())
    print(model[0].dtype, 'size = ', model[0].element_size(), 'Bytes')
    print(len(model))
    print('len =', model[0].shape[0])
    print('memory = {:.2f} '.format(model[0].element_size() * model[0].shape[0] / 1e6), 'MBytes\n')

# load_model('/Users/jindajia/Downloads/data/params/params_iter_001000/000.pt')
# load_model('/Users/jindajia/Downloads/data/params/params_iter_001000/001.pt')
# load_model('/Users/jindajia/Downloads/data/params/params_iter_002000/000.pt')
# load_model('/Users/jindajia/Downloads/data/params/params_iter_002000/001.pt')
#
# load_model('/Users/jindajia/Downloads/data/grads/grads_iter_001000/000.pt')
# load_model('/Users/jindajia/Downloads/data/grads/grads_iter_001000/001.pt')
# load_model('/Users/jindajia/Downloads/data/grads/grads_iter_002000/000.pt')
# load_model('/Users/jindajia/Downloads/data/grads/grads_iter_002000/001.pt')

# t1 = load_model('/Users/jindajia/Developer/log/module.encoder.encoder.layer.4.intermediate.dense.weight_params_step_100_GPU_0.pt')
# t2 = load_model('/Users/jindajia/Developer/log/module.encoder.encoder.layer.4.intermediate.dense.weight_params_step_100_GPU_1.pt')
# print(torch.equal(t1, t2))  # 输出 True 或 False

load_model('/Users/jindajia/Developer/log/bert_pretrain.2023.8.30.11.22.6.addjtvxg/bert_pretrain.2023.8.30.11.22.6.addjtvxg/global_step1000/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt')

