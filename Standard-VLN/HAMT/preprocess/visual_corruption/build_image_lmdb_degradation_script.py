import os
import sys
import numpy as np
import json
import collections
from PIL import Image
import lmdb
import math
import argparse
import dotenv

dotenv.load_dotenv("../../../.env")

import image_degradations

sys.path.append(os.getenv("MATTERSIM_PATH"))
import MatterSim

from tqdm import tqdm

NOISE_MODELS = collections.OrderedDict()
NOISE_MODELS["defocus_blur"] = image_degradations.defocus_blur
# NOISE_MODELS["motion_blur"] = image_degradations.motion_blur
NOISE_MODELS["lighting"] = image_degradations.lighting
NOISE_MODELS["speckle_noise"] = image_degradations.speckle_noise
NOISE_MODELS["spatter"] = image_degradations.spatter

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60

scan_data_dir = os.getenv('SCANS_PATH')
connectivity_dir = '../../datasets/R2R/connectivity'

TEST_MODE = False

NEWHEIGHT = 248
NEWWIDTH = int(WIDTH / HEIGHT * NEWHEIGHT)
print(NEWHEIGHT, NEWWIDTH)

def setup_simulator():
    sim = MatterSim.Simulator()
    sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setPreloadingEnabled(True)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def load_viewpoints(sim):
    viewpoint_ids = []
    print(os.path.join(connectivity_dir, 'scans.txt'))
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan_idx, scan in enumerate(scans):
        # if TEST_MODE and scan_idx >5:
        #     break #TODO: remove
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids


def perform_degradation(image, corruption, severity=3):
    degradation_method = lambda clean_image: NOISE_MODELS[corruption](clean_image, severity)
    degraded_img = np.uint8(degradation_method(image))
    return degraded_img

def save_image(image, op_dir, pretext=''):
    os.makedirs(op_dir, exist_ok=True)
    img_path = os.path.join(op_dir, f'{pretext}.png' )
    Image.fromarray(image).save(img_path)


def main():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--output_dir', type=str, default='../../../panos', help='path to save lmdb directory')
    parser.add_argument('--visual_degradation', default='na', choices=['lighting', 'spatter', 'motion_blur', 'defocus_blur', 'speckle_noise'])

    args = parser.parse_args()

    degradation_type = args.visual_degradation
    lmdb_file_name = f'{degradation_type}/panoimages.lmdb'
    lmdb_path = os.path.join(args.output_dir, lmdb_file_name) 
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    sim = setup_simulator()

    viewpoint_ids = load_viewpoints(sim)

    data_size_per_img = np.random.randint(255, size=(NEWHEIGHT, NEWWIDTH, 3), dtype=np.uint8).nbytes
    print(data_size_per_img, 36*data_size_per_img*len(viewpoint_ids))

    env = lmdb.open(lmdb_path, map_size=int(1e12))

    for i, viewpoint_id in tqdm(enumerate(viewpoint_ids), total=len(viewpoint_ids), desc='Building degraded pano lmdb'):
        scan, vp = viewpoint_id
        if i % 100 == 0:
            print(i, scan, vp)
        print(i, scan, vp)
        key = '%s_%s' % (scan, vp)
        key_byte = key.encode('ascii')
        
        print(key_byte)

        txn = env.begin(write=True)
        
        images = []
        for ix in range(36):
            if ix == 0:
                sim.newEpisode([scan], [vp], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix
            image = np.array(state.rgb, copy=True) # in BGR channel
            image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # resize
            image = image.resize((NEWWIDTH, NEWHEIGHT), Image.ANTIALIAS)
            image = np.array(image)
            
            # save_image(image, op_dir='test_build_pano_degraded_arg', pretext=key+"_before_"+str(ix))
            image_degraded = perform_degradation(image, degradation_type)
            # save_image(image_degraded, op_dir='test_build_pano_degraded_arg', pretext=key+"_after_" + degradation_type + "_" + str(ix))

            images.append(image_degraded)
        images = np.stack(images, 0)
        images_c = images.copy(order='C')
        txn.put(key_byte, images_c)
        txn.commit()
    env.close()
    print(f'Saved lmdb to {lmdb_path}.')


if __name__ == '__main__':
    main()