import pytorch2timeloop
import os
import yaml
import json
import copy
import torch
import torch.nn as nn

from torchprofile import profile_macs
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm
from pathlib import Path
from datetime import date


def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=False, activation_bits=32):
    act_byte = activation_bits / 8
    model = copy.deepcopy(net)

    # noinspection PyArgumentList
    def count_convNd(m, x, y):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

    # noinspection PyArgumentList
    def count_linear(m, x, y):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_bn(m, x, _):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_relu(m, x, _):
        # count activation size required by backward
        if require_backward:
            m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_smooth_act(m, x, _):
        # count activation size required by backward
        if require_backward:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('grad_activations', torch.zeros(1))
        m_.register_buffer('tmp_activations', torch.zeros(1))

        if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            fn = count_convNd
        elif type(m_) in [nn.Linear]:
            fn = count_linear
        elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
            fn = count_bn
        elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
            fn = count_relu
        elif type(m_) in [nn.Sigmoid, nn.Tanh]:
            fn = count_smooth_act
        else:
            fn = None

        if fn is not None:
            _handler = m_.register_forward_hook(fn)

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(model.parameters().__next__().device)
    with torch.no_grad():
        model(x)

    memory_info_dict = {
        'peak_activation_size': torch.zeros(1),
        'residual_size': torch.zeros(1),
    }

    for m in model.modules():
        if len(list(m.children())) == 0:
            def new_forward(_module):
                def lambda_forward(_x):
                    current_act_size = _module.tmp_activations + memory_info_dict['residual_size']
                    memory_info_dict['peak_activation_size'] = max(
                        current_act_size, memory_info_dict['peak_activation_size']
                    )
                    return _module.old_forward(_x)

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

        if type(m) in [BasicBlock]:
            def new_forward(_module):
                def lambda_forward(_x):
                    memory_info_dict['residual_size'] = _x.numel() * act_byte
                    result = _module.old_forward(_x)
                    memory_info_dict['residual_size'] = 0
                    return result

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

    with torch.no_grad():
        model(x)

    return memory_info_dict['peak_activation_size'].item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), require_backward=False,
                        activation_bits=32, batch_size=1):
    activation_size = count_activation_size(net, input_size, require_backward, activation_bits)

    memory_cost = activation_size * batch_size
    return memory_cost


class Profiler(object):
    def __init__(self,
                 net_name,
                 arch_name,
                 model,
                 input_size,
                 batch_size,
                 convert_fc,
                 exception_module_names=None
                 ):
        self.base_dir = Path(os.getcwd())
        self.sub_dir = date.today().strftime("%b-%d-%Y")
        self.top_dir = '/'.join(['networks', net_name])
        self.model = model
        self.timeloop_dir = '/'.join(['archs', arch_name])
        self.arch_name = arch_name
        self.net_name = net_name
        self.input_size = input_size
        self.batch_size = batch_size
        self.convert_fc = convert_fc
        self.exception_module_names = exception_module_names

        self.profiled_lib_dir = '/'.join(['.', self.timeloop_dir, 'profiled_lib.json'])
        self.profiled_lib = {}
        self.load_profiled_lib()

    def load_profiled_lib(self):
        if os.path.exists(self.profiled_lib_dir):
            with open(self.profiled_lib_dir, 'r') as fid:
                self.profiled_lib = json.load(fid)

    def write_profiled_lib(self):
        with open(self.profiled_lib_dir, 'w') as fid:
            json.dump(self.profiled_lib, fid, sort_keys=True)

    def profile(self):
        # convert the model to timeloop files
        pytorch2timeloop.convert_model(
            self.model,
            self.input_size,
            self.batch_size,
            self.sub_dir,
            self.top_dir,
            self.convert_fc,
            self.exception_module_names
        )
        layer_dir = self.base_dir/self.top_dir/self.sub_dir
        if not os.path.exists(layer_dir):
            p = Path(layer_dir)
            p.mkdir(parents=True, exist_ok=True)

        # check duplicated layer info
        layer_info = {}
        path, dirs, files = next(os.walk(layer_dir))
        file_count = len(files)
        for idx in range(file_count):
            file = layer_dir/f"{self.sub_dir}_layer{idx + 1}.yaml"
            with open(file, 'r') as fid:
                layer_dict = yaml.safe_load(fid)
                for layer_id, info in layer_info.items():
                    if info['layer_dict'] == layer_dict:
                        layer_info[layer_id]['num'] += 1
                        break
                else:
                    layer_info[idx+1] = {
                        'layer_dict': layer_dict,
                        'num': 1,
                        'name': str(file).replace('.yaml', '')
                    }

        # check the mapper info
        with open(self.base_dir/self.timeloop_dir/'mapper/mapper.yaml', 'r') as fid:
            mapper_dict = yaml.safe_load(fid)
            for layer_id, info in layer_info.items():
                layer_info[layer_id]['mapper_timeout'] = mapper_dict['mapper']['timeout']
                layer_info[layer_id]['mapper_algo'] = mapper_dict['mapper']['algorithm']
                layer_info[layer_id]['mapper_victory_condition'] = mapper_dict['mapper']['victory-condition']
                layer_info[layer_id]['mapper_max_permutations'] = mapper_dict['mapper']['max-permutations-per-if-visit']

        # check whether some layers have been profiled before and exist in the
        # profiled_lib.
        # sometimes the layer_dict are the same but name will be different, in
        # that case make their name the same
        for layer_id, info in layer_info.items():
            for profiled_name, profiled_info in self.profiled_lib.items():
                if info['layer_dict'] == profiled_info['layer_dict'] and \
                        info['mapper_timeout'] == profiled_info['mapper_timeout'] and \
                        info['mapper_algo'] == profiled_info['mapper_algo'] and \
                        info['mapper_victory_condition'] == profiled_info['mapper_victory_condition'] and \
                        info['mapper_max_permutations'] ==  profiled_info['mapper_max_permutations']:

                    print(f"Found layer {layer_id} in the profiled library of layers")
                    layer_info[layer_id]['energy'] = profiled_info['energy']
                    layer_info[layer_id]['area'] = profiled_info['area']
                    layer_info[layer_id]['cycle'] = profiled_info['cycle']
                    layer_info[layer_id]['name'] = profiled_name

        # run timeloop
        print(f'running timeloop to get energy and latency...')
        
        timeloop_layer_dir = self.base_dir/self.timeloop_dir/'profiled_networks'/self.net_name/self.sub_dir
        # if not os.path.exists(timeloop_layer_dir):
        #     p = Path(timeloop_layer_dir)
        #     p.mkdir(parents=True, exist_ok=True)
            
        for layer_id in layer_info.keys():
            os.makedirs(timeloop_layer_dir + '/' + f'layer{layer_id}', exist_ok=True)

        def get_cmd(layer_id):
            cwd = timeloop_layer_dir + '/' + f'layer{layer_id}'
            if 'M' in layer_info[layer_id]['layer_dict']['problem']['instance']:
                constraint_pth = self.base_dir/self.timeloop_dir/'constraints/*.yaml'
            else:
                # depthwise
                constraint_pth = self.base_dir/self.timeloop_dir/'constraints_dw/*.yaml'

            timeloopcmd = f"timeloop-mapper " \
                          f"{self.base_dir/self.timeloop_dir/'arch'}" + "/" + f"{self.arch_name}.yaml " \
                          f"{self.base_dir/self.timeloop_dir/'arch/components/*.yaml'} " \
                          f"{self.base_dir/self.timeloop_dir/'mapper/mapper.yaml'} " \
                          f"{constraint_pth} " \
                          f"{self.base_dir/self.top_dir/self.sub_dir/self.sub_dir}_layer{layer_id}.yaml "
            return [cwd, timeloopcmd]

        cmds_list = []
        for layer_id in layer_info.keys():
            if 'energy' in layer_info[layer_id].keys():
                # the layer is in the profiler lib
                continue
            else:
                cmds_list.append(get_cmd(layer_id))

        for cwd, cmd in tqdm(cmds_list):
            os.chdir(cwd)
            os.system(cmd)
        os.chdir(self.base_dir)

        print(f'timeloop running finished!')

        for layer_id in layer_info.keys():
            if 'energy' in layer_info[layer_id].keys():
                # the layer is in the profiler lib
                continue
            with open('/'.join([timeloop_layer_dir, f'layer{layer_id}', f'timeloop-mapper.stats.txt']), 'r') as fid:
                lines = fid.read().split('\n')[-200:]
                for line in lines:
                    if line.startswith('Total topology energy'):
                        energy = line.split(': ')[1].split(' ')[0]
                        layer_info[layer_id]['energy'] = eval(energy)
                    elif line.startswith('Total topology area'):
                        area = line.split(': ')[1].split(' ')[0]
                        layer_info[layer_id]['area'] = eval(area)
                    elif line.startswith('Max topology cycles'):
                        cycle = line.split(': ')[1]
                        layer_info[layer_id]['cycle'] = eval(cycle)

        for layer_id in layer_info.keys():
            layer_name = layer_info[layer_id]['name']
            if layer_name not in self.profiled_lib.keys():
                info = {
                    'layer_dict': layer_info[layer_id]['layer_dict'],
                    'energy': layer_info[layer_id]['energy'],
                    'area': layer_info[layer_id]['area'],
                    'cycle': layer_info[layer_id]['cycle'],
                    'mapper_timeout': layer_info[layer_id]['mapper_timeout'],
                    'mapper_algo': layer_info[layer_id]['mapper_algo'],
                    'mapper_victory_condition': layer_info[layer_id]['mapper_victory_condition'],
                    'mapper_max_permutations': layer_info[layer_id]['mapper_max_permutations']
                }
                self.profiled_lib[layer_name] = info

        self.write_profiled_lib()

        overall = {}
        total_energy = 0
        total_cycle = 0

        for layer_id, info in layer_info.items():
            total_energy += info['energy'] * info['num']
            total_cycle += info['cycle'] * info['num']

        overall['total_energy'] = total_energy
        overall['total_cycle'] = total_cycle
        overall['num_params'] = sum(p.numel() for p in
                                    self.model.parameters() if
                                    p.requires_grad)
        #overall['macs'] = profile_macs(self.model, torch.randn([1] + list(self.input_size)))
        #overall['activation_size'] = count_activation_size(self.model, [1] + list(self.input_size))

        return layer_info, overall