import os
import time
from tqdm import tqdm

from utils.misc import DATASET_TO_MAX_INSTRUCTIONS

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.results = {}
        self.losses = [] # For learning agents

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, max_processed=1000000, **kwargs):
        max_instr = DATASET_TO_MAX_INSTRUCTIONS[self.args.dataset.lower()]

        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in tqdm(range(iters), total=iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            processed = 0
            while True:
                time0 = time.time()
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                        processed += 1
                        
                        with open(os.path.join(self.args.log_dir, "valid.txt"), 'a') as f:
                            f.write(f"Processed {processed:04d}/{min(max_processed, max_instr)} instructions | Time elapsed: {time.time() - time0:.2f}\n")
                        
                    if processed >= max_processed:
                        looped = True
                        break
                if looped:
                    break


