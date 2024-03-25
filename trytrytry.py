import torch
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

# 创建一个简单的3D球体
sphere_mesh = ico_sphere(3, device=torch.device("cuda"))

# 打印球体的顶点和面信息
print("顶点坐标：", sphere_mesh.verts_packed())
print("面索引：", sphere_mesh.faces_packed())
