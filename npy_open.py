import numpy as np
import open3d as o3d

# .npy 파일 로드
point_cloud = np.load('3d_mod_av_db\\TRAIN_SET\\points\\00000001.npy')

# x, y, z 좌표만 사용
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # 첫 세 열은 x, y, z 좌표

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd])