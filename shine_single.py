import sys
from numpy.linalg import inv, norm
from tqdm import tqdm
import torch

from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset


def run_shine_mapping_single():
        
    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit(
            "Please provide the path to the config file.\nTry: python shine_batch.py xxx/xxx_config.yaml"
        )
    
    run_path = setup_experiment_and_return_run_path(config)
    dev = config.device

    # for each frame
    print("Load, preprocess and sample data")
    
    for frame_id in range(config.begin_frame, config.end_frame):
        if (frame_id % config.every_frame != 0): 
            continue
        
        print("processing frame id = ", frame_id)
        
        # initialize the mlp decoder
        geo_mlp = Decoder(config, is_geo_encoder=True)
    
        # initialize the feature octree
        octree = FeatureOctree(config)
        
        # dataset
        dataset = LiDARDataset(config, octree)
        
        mesher = Mesher(config, octree, geo_mlp, None)
        mesher.global_transform = inv(dataset.begin_pose_inv)
        
        t0 = get_time()
        # preprocess, sample data and update the octree
        dataset.process_single_frame(frame_id)
        t1 = get_time()
        print("data preprocessing and sampling time (s): %.3f" %(t1 - t0))
        
        # 保存点云
        save_pc = False
        if save_pc:
            pc_map_path = run_path + '/map/pc_map_down.ply'
            dataset.write_merged_pc(pc_map_path)

        # learnable parameters
        octree_feat = list(octree.parameters())
        geo_mlp_param = list(geo_mlp.parameters())

        # learnable sigma for differentiable rendering
        sigma_size = torch.nn.Parameter(torch.ones(1, device=dev)*1.0) 
        # fixed sigma for sdf prediction supervised with BCE loss
        sigma_sigmoid = config.logistic_gaussian_ratio * config.sigma_sigmoid_m * config.scale

        # initialize the optimizer
        opt = setup_optimizer(config, octree_feat, geo_mlp_param, None, sigma_size)

        octree.print_detail()

        # begin training
        print("Begin mapping")
        cur_base_lr = config.lr
        for iter in tqdm(range(config.iters)): # 默认迭代 20,000 次
            # learning rate decay
            step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)
            
            # load batch data (avoid using dataloader because the data are already in gpu, memory vs speed)
            # loss computed based on each point sample  
            coord, sdf_label, normal_label, _, weight = dataset.get_batch()

            if config.normal_loss_on or config.ekional_loss_on:
                coord.requires_grad_(True)

            feature = octree.query_feature(coord) # interpolate and concat the hierachical grid features    

            pred = geo_mlp.sdf(feature) # predict the scaled sdf with the feature

            surface_mask = weight > 0
            cur_loss = 0.
            # calculate the loss

            weight = torch.abs(weight) # weight's sign indicate the sample is around the surface or in the free space
           
            sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction) 
                
            cur_loss += sdf_loss
            
            # optional loss (ekional, normal loss)
            if config.normal_loss_on or config.ekional_loss_on:
                g = get_gradient(coord, pred)*sigma_sigmoid
            eikonal_loss = 0.
            if config.ekional_loss_on:
                eikonal_loss = ((g[surface_mask].norm(2, dim=-1) - 1.0) ** 2).mean() # MSE with regards to 1  
                cur_loss += config.weight_e * eikonal_loss
            normal_loss = 0.
            if config.normal_loss_on:
                g_direction = g / g.norm(2, dim=-1)
                normal_diff = g_direction - normal_label
                normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean() 
                cur_loss += config.weight_n * normal_loss

            opt.zero_grad(set_to_none=True)
            cur_loss.backward()
            opt.step()

            # reconstruction by marching cubes
            if (((iter+1) % config.vis_freq_iters) == 0 and iter > 0): 
                print("Begin mesh reconstruction from the implicit map")               
                mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + "_frameID_" + str(frame_id) + ".ply"

                mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, None, config.save_map, config.semantic_on)
    

if __name__ == "__main__":
    run_shine_mapping_single()
