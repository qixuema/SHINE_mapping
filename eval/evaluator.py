import csv
from eval_utils import eval_mesh

########################################### MaiCity Dataset ###########################################
dataset_name = "maicity_01_"

# ground truth point cloud (or mesh) file
# (optional masked by the intersection part of all the compared method)
# gt_pcd_path = "xxx/maicity/01/gt_map_pc_mai.ply"
gt_pcd_path = "/home/ubuntu/studio/project/shine/SHINE_mapping/experiments/maicity_batch_2022-12-30_15-05-31/map/pc_map_down.ply"


# pred_mesh_path = "/home/ubuntu/studio/project/shine/SHINE_mapping/experiments/maicity_batch_2022-12-30_15-05-31/mesh/mesh_iter_20000.ply"
pred_mesh_path = "/home/ubuntu/studio/project/shine/SHINE_mapping/experiments/undc/maicity/mai-ori.ply"

method_name = "ours_mai"

# pred_mesh_path = "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "vdb_fusion_xxx"

# pred_mesh_path = "xxx/baseline/puma_xxx.ply"
# method_name = "puma_xxx"

########################################### MaiCity Dataset ###########################################


######################################## Newer College Dataset ########################################
# dataset_name = "ncd_quad_"

# gt_pcd_path = "/home/ubuntu/studio/project/shine/SHINE_mapping/experiments/ncd_batch_2022-12-30_11-45-18/map/pc_map_down.ply"

# pred_mesh_path = "/home/ubuntu/studio/project/shine/SHINE_mapping/experiments/ncd_batch_2022-12-30_11-45-18/mesh/mesh_iter_20000.ply"
# pred_mesh_path = "/home/ubuntu/studio/project/shine/SHINE_mapping/experiments/undc/ncd/ncd-reori.ply"

# method_name = "ours_ncd"

# pred_mesh_path = "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "vdb_fusion_xxx"

# pred_mesh_path = "xxx/baseline/puma_xxx.ply"
# method_name = "puma_xxx"

######################################## Newer College Dataset ########################################

# evaluation results output file
base_output_folder = "./experiments/evaluation/"

output_csv_path = base_output_folder + dataset_name + method_name + "_eval.csv"

# evaluation parameters
# For MaiCity
down_sample_vox = 0.01
dist_thre = 0.1
truncation_dist_acc = 0.2 
truncation_dist_com = 2.0

# For NCD
# down_sample_vox = 0.02
# dist_thre = 0.2
# truncation_dist_acc = 0.4
# truncation_dist_com = 2.0

# evaluation
eval_metric = eval_mesh(pred_mesh_path, 
                        gt_pcd_path, 
                        down_sample_res=down_sample_vox, 
                        threshold=dist_thre, 
                        truncation_acc = truncation_dist_acc, 
                        truncation_com = truncation_dist_com, 
                        gt_bbx_mask_on = True) 

print(eval_metric)

evals = [eval_metric]

csv_columns = ['MAE_accuracy (m)', 'MAE_completeness (m)', 'Chamfer_L1 (m)', 'Chamfer_L2 (m)', \
        'Precision [Accuracy] (%)', 'Recall [Completeness] (%)', 'F-score (%)', 'Spacing (m)', \
        'Inlier_threshold (m)', 'Outlier_truncation_acc (m)', 'Outlier_truncation_com (m)']

try:
    with open(output_csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in evals:
            writer.writerow(data)
except IOError:
    print("I/O error")

