import open3d as o3d
path = "/home/ubuntu/studio/project/blender/data/fusion/MultiFrame/mesh/mf_"

for i in range(32,33):
    path_input  = path + str(i).zfill(4) + "_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(path_input)
    path_output = path + str(i).zfill(4) + "_mesh.obj"
    o3d.io.write_triangle_mesh(path_output, mesh, write_triangle_uvs=True)