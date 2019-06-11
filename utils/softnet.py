import os
import torch
import numpy as np

def filter_dict(full_dict, blacklist):
    sheeps = list(filter(
        lambda k: any(b in k for b in blacklist),
        full_dict.keys()))
    for sheep in sheeps:
        full_dict.pop(sheep)
    return full_dict

class SoftUpdateNetwork:
    def __init__(self, model_path, save, load, delay):
        self.save = save
        self.load = load
        self.delay = delay
        self.model_path = model_path

        self.count = 0

    def share_memory(self):
        self.ac_target.share_memory()
        self.ac_explorer.share_memory()
        #  print(self.ac_explorer.state_dict().keys())

    def sync_target(self, i, blacklist):
        self.load_target(i, "eac", blacklist)
    def sync_explorer(self, i, blacklist):
        self.load_explorer(i, "eac", blacklist)

    def sync_master(self, blacklist):
        self.load_models(0, "eac", blacklist) # help apha actor to explore ~ multi agent coop !!
        #  for explorer in self.ac_explorer.actor: # load only alpha actor, keep critics our own, and encoder is shared ...
        #      self._soft_update(self.ac_target.actor[0].parameters(), explorer, 1.)

    def _soft_update_mean(self, explorers, targets, tau):
        if not tau:
            return

        params = np.mean([list(explorer.parameters()) for explorer in explorers], 0)
        self._soft_update(params, targets, tau)

    def _soft_update(self, explorer_params, targets, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(targets.parameters(), explorer_params):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)

    def save_models(self, model_id, prefix):
        if not self.save:
            return
        self.count += 1
        if self.count % self.delay:
            return
        self.save_models_ex(model_id, prefix)

    def save_models_ex(self, model_id, prefix):
        target = os.path.join(self.model_path, '%s_target_%s'%(prefix, model_id))
        torch.save(self.ac_target.state_dict(), target)

        explorer = os.path.join(self.model_path, '%s_explorer_%s'%(prefix, model_id))
        torch.save(self.ac_explorer.state_dict(), explorer)

    def load_models(self, model_id, prefix, blacklist = []):
        self.load_target(model_id, prefix, blacklist)
        self.load_explorer(model_id, prefix, blacklist)

    def load_explorer(self, model_id, prefix, blacklist = []):
        if not self.load:
            return
        path = os.path.join(self.model_path, '%s_explorer_%s'%(prefix, model_id))
        if not os.path.exists(path):
            return
        model = filter_dict(torch.load(path), blacklist)
        self.ac_explorer.load_state_dict(model, strict=False)

    def load_target(self, model_id, prefix, blacklist = []):
        if not self.load:
            return
        target = os.path.join(self.model_path, '%s_target_%s'%(prefix, model_id))
        if not os.path.exists(target):
            return
        model = filter_dict(torch.load(target), blacklist)
        self.ac_target.load_state_dict(model, strict=False)
