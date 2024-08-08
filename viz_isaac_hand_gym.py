
from envs.gym_envs import IsaacHandVisualizerEnv
from avp_stream import VisionProStreamer
import time 
from typing import * 
import numpy as np 
import torch 

avp_ip = "192.168.123.253"  
class IsaacVisualizer:

    def __init__(self, args): 
        self.env = IsaacHandVisualizerEnv(args)
        self.s = VisionProStreamer(args.ip, record = True)
    def run(self):

        while True: 
            latest = self.s.latest
            self.env.step(np2tensor(latest, self.env.device)) 



def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:  
    for key in data.keys():
        data[key] = torch.tensor(data[key], dtype = torch.float32, device = device)
    return data


if __name__ == "__main__":
    import argparse 
    import os 
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--follow', action = 'store_true', help = "The viewpoint follows the users head")
    parser.add_argument('--ip',default =  avp_ip,type = str, required = False)
    args = parser.parse_args()

    vis = IsaacVisualizer(args)
    vis.run()