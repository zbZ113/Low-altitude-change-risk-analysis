import torch
import torch.optim as optim

def create_optimizer(model, lr, weight_decay):
    param_groups = [
        {'params': model.encoder.parameters(),        'lr': lr},        # was lr*0.1
        {'params': model.change_decoder.parameters(),'lr': lr},
        {'params': model.sem_decoder_t1.parameters(),'lr': lr},
        {'params': model.sem_decoder_t2.parameters(),'lr': lr},
        {'params': model.cross_attn.parameters(),    'lr': lr * 0.9},   # was 2x
        {'params': model.residual_fusion.parameters(),'lr': lr * 0.9}
    ]
    return optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


class EMA:
    """指数移动平均模型"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + \
                              self.decay * self.shadow.get(name, param.data)
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def setup_ema(model, decay=0.999):
    ema = EMA(model, decay)
    ema.register()
    ema.model = model  # 统一访问模型用法
    return ema
