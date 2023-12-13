# projects/Meg_dataset/mmdet3d/models/lss_transform_simple.py


# from mmdet3d.models.vtransforms.lss import LSSTransform

from mmdet3d.registry import MODELS

from .depth_lss import LSSTransform

__all__ = ["LSSTransformSimple"]

# @MODELS.register_module()
# class LSSTransform(BaseViewTransform):

@MODELS.register_module()
class LSSTransformSimple(LSSTransform):
    """
    原版的 LSSTransform 继承 BaseTransform， BaseTransform 在 forward 里传入了一堆没用的参数。
    这里将其重写，只保留有用的参数。
    """
    def forward(
        self,
        img,
        camera2lidar,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x
