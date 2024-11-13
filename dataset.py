"""
    This code is adapted from https://github.com/aharley/simple_bev/nuscenesdataset.py
"""

import numpy as np
import torch

import os
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box

import torchvision
from torch.utils.data import Dataset
from functools import reduce
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import transform_matrix


from models.bev_module.utils import geom, py, vox


discard_invisible = False

totorch_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))

def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img

def move_refcam(data, refcam_id):

    data_ref = data[refcam_id].clone()
    data_0 = data[0].clone()

    data[0] = data_ref
    data[refcam_id] = data_0

    return data

def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

class LidarPointCloud(PointCloud):
    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 5

    @classmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """
        
        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)
        
        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)

def get_lidar_data(nusc, sample_rec, nsweeps, min_distance, dataroot):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # points = np.zeros((5, 0))
    points = np.zeros((6, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # print('time_lag', time_lag)
        # print('new_points', new_points.shape)
        
        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

def get_radar_data(nusc, sample_rec, nsweeps, min_distance, use_radar_filters, dataroot):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt, ring_index) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # import ipdb; ipdb.set_trace()
    
    # points = np.zeros((5, 0))
    points = np.zeros((19, 0)) # 18 plus one for time

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['RADAR_FRONT']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),inverse=True)

    if use_radar_filters:
        RadarPointCloud.default_filters()
    else:
        RadarPointCloud.disable_filters()

    # Aggregate current and previous sweeps.
    # from all radars 
    radar_chan_list = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for radar_name in radar_chan_list:
        sample_data_token = sample_rec['data'][radar_name]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = RadarPointCloud.from_file(os.path.join(dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # print('time_lag', time_lag)
            # print('new_points', new_points.shape)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class NuscenesDataset(Dataset):

    def __init__(
        self,
        args,
        nusc
    ):
        self.is_train = args.is_train

        print('is_train', self.is_train)
        #self.crop_offset = args.crop_offset
        self.rand_crop_and_resize = args.rand_crop_and_resize
        self.final_dim = args.resize_to
        if self.rand_crop_and_resize:
            self.resize_lim = [0.8,1.2]
            self.crop_offset = int(self.final_dim[0]*(1-self.resize_lim[0]))
        else:
            self.resize_lim  = [1.0,1.0]
            self.crop_offset = 0
        self.H = args.H
        self.W = args.W
        self.cams = args.cams
        self.ncams = args.ncams
        self.do_shuffle_cams = args.do_shuffle_cams
        self.refcam_id = args.refcam_id
        self.bounds = args.bounds

        self.version = args.version

        self.dataroot = args.dataset_path

        self.X, self.Y, self.Z = args.voxel_size
        self.nusc = nusc

        split = 'train' if args.is_train else 'val'
        scenes = create_splits_scenes()[split]

        samples = [samp for samp in self.nusc.sample]
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in scenes]
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        
        self.samples = samples

        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = args.bounds

        grid_conf = { # note the downstream util uses a different XYZ ordering
            'xbound': [XMIN, XMAX, (XMAX-XMIN)/float(self.X)],
            'ybound': [ZMIN, ZMAX, (ZMAX-ZMIN)/float(self.Z)],
            'zbound': [YMIN, YMAX, (YMAX-YMIN)/float(self.Y)],
        }
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.nsweeps = 1
        self.use_radar_filters = False

        # the scene centroid is defined wrt a reference camera,
        # which is usually random
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 # down 1 meter
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float().cuda()

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=scene_centroid,
            bounds=self.bounds,
            assert_cube=False)
        
        self.num_queries = args.num_queries

        self.get_sem_masks = args.get_sem_masks
        
    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        cams = self.choose_cams()
        refcam_id = self.choose_ref_cam()

        imgs, rots, trans, intrins = self.get_image_data(sample, cams)

        # move the target refcam_id to the zeroth slot
        imgs = move_refcam(imgs, refcam_id)
        rots = move_refcam(rots, refcam_id)
        trans = move_refcam(trans, refcam_id)
        intrins = move_refcam(intrins, refcam_id)

        binimg, egopose, multibinimg = self.get_binimg(sample)

        lidar_data = self.get_lidar_data(sample, nsweeps=self.nsweeps)
        radar_data = self.get_radar_data(sample, nsweeps=self.nsweeps)

        lidar_extra = lidar_data[3:]
        lidar_data = lidar_data[:3]

        lrtlist_, boxlist_, vislist_, tidlist_ = self.get_lrtlist(sample)

        N_ = lrtlist_.shape[0]
        

        if N_ > 0:
            
            velo_T_cam = geom.merge_rt(rots, trans)
            cam_T_velo = geom.safe_inverse(velo_T_cam)

            # note we index 0:1, since we already put refcam into zeroth position
            lrtlist_cam = geom.apply_4x4_to_lrt(cam_T_velo[0:1].repeat(N_, 1, 1), lrtlist_).unsqueeze(0)

            seg_bev, valid_bev, multi_seg_bev, multi_valid_bev, xz_centers = self.get_seg_bev(lrtlist_cam, vislist_)
            
            center_bev, offset_bev, size_bev, ry_bev, ycoord_bev = self.get_center_and_offset_bev(lrtlist_cam, seg_bev)

            translation_rotation_list = []
            assert len(boxlist_) == len(xz_centers), f'len(boxlist_)={len(boxlist_)} len(xz_centers)={len(xz_centers)}'

            for i, box_ in enumerate(boxlist_):
                translation_rotation_list.append(torch.Tensor([xz_centers[i][0], xz_centers[i][1], xz_centers[i][2], xz_centers[i][3]])) #, box_[7]]))   #[box_[0], box_[1], box_[2], box_[7]]))

            translation_rotation_list = torch.cat([i.unsqueeze(0) for i in translation_rotation_list], dim=0)     # (num_vechicels, 4)

        
        else:
            # no vehicles in this sample
            seg_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            multi_seg_bev = torch.zeros((0, self.Z, self.X), dtype=torch.float32)
            valid_bev = torch.ones((1, self.Z, self.X), dtype=torch.float32)
            multi_valid_bev = torch.ones((0, self.Z, self.X), dtype=torch.float32)
            center_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            offset_bev = torch.zeros((2, self.Z, self.X), dtype=torch.float32)
            size_bev = torch.zeros((3, self.Z, self.X), dtype=torch.float32)
            ry_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            ycoord_bev = torch.zeros((1, self.Z, self.X), dtype=torch.float32)
            translation_rotation_list = torch.zeros((0, 4), dtype=torch.float32)
        
        assert len(translation_rotation_list) == len(multi_seg_bev), f'len(translation_rotation_list)={len(translation_rotation_list)} len(multi_seg_bev)={len(multi_seg_bev)}'
        #assert translation_rotation_list.shape[1] == 3, f'translation_rotation_list.shape[1]={translation_rotation_list.shape[1]}'

        N = 150 # i've seen n as high as 103 before, so 150 is probably safe (max number of objects)
        lrtlist = torch.zeros((N, 19), dtype=torch.float32)
        vislist = torch.zeros((N), dtype=torch.float32)
        scorelist = torch.zeros((N), dtype=torch.float32)
        lrtlist[:N_] = lrtlist_
        vislist[:N_] = vislist_
        scorelist[:N_] = 1

        # lidar is shaped 3,V, where V~=26k 
        times = lidar_extra[2] # V
        inds = times==times[0]
        lidar0_data = lidar_data[:,inds]
        lidar0_extra = lidar_extra[:,inds]

        lidar0_data = np.transpose(lidar0_data)
        lidar0_extra = np.transpose(lidar0_extra)
        lidar_data = np.transpose(lidar_data)
        lidar_extra = np.transpose(lidar_extra)
        V = 30000*self.nsweeps
            
        if lidar_data.shape[0] > V:
            # assert(False) # if this happens, it's probably better to increase V than to subsample as below
            lidar0_data = lidar0_data[:V//self.nsweeps]
            lidar0_extra = lidar0_extra[:V//self.nsweeps]
            lidar_data = lidar_data[:V]
            lidar_extra = lidar_extra[:V]
        elif lidar_data.shape[0] < V:
            lidar0_data = np.pad(lidar0_data,[(0,V//self.nsweeps-lidar0_data.shape[0]),(0,0)],mode='constant')
            lidar0_extra = np.pad(lidar0_extra,[(0,V//self.nsweeps-lidar0_extra.shape[0]),(0,0)],mode='constant')
            lidar_data = np.pad(lidar_data,[(0,V-lidar_data.shape[0]),(0,0)],mode='constant')
            lidar_extra = np.pad(lidar_extra,[(0,V-lidar_extra.shape[0]),(0,0)],mode='constant',constant_values=-1)
        lidar0_data = np.transpose(lidar0_data)
        lidar0_extra = np.transpose(lidar0_extra)
        lidar_data = np.transpose(lidar_data)
        lidar_extra = np.transpose(lidar_extra)

        # radar has <700 points 
        radar_data = np.transpose(radar_data)
        V = 700*self.nsweeps
        if radar_data.shape[0] > V:
            # print('radar_data', radar_data.shape)
            # print('max pts', V)
            assert(False) # i expect this to never happen
            radar_data = radar_data[:V]
        elif radar_data.shape[0] < V:
            radar_data = np.pad(radar_data,[(0,V-radar_data.shape[0]),(0,0)],mode='constant')
        radar_data = np.transpose(radar_data)

        lidar0_data = torch.from_numpy(lidar0_data).float()
        lidar0_extra = torch.from_numpy(lidar0_extra).float()
        lidar_data = torch.from_numpy(lidar_data).float()
        lidar_extra = torch.from_numpy(lidar_extra).float()
        radar_data = torch.from_numpy(radar_data).float()

        binimg = (binimg > 0).float()
        seg_bev = (seg_bev > 0).float()
        multi_seg_bev = (multi_seg_bev > 0).float()

        dataset_dict = {}
    
        dataset_dict["images"] = imgs               # (ncams, 3, H, W)
        dataset_dict["rots"] = rots
        dataset_dict["trans"] = trans
        dataset_dict["intrins"] = intrins
        dataset_dict["seg_bev"] = seg_bev
        dataset_dict["valid_bev"] = valid_bev
        dataset_dict["center_bev"] = center_bev     # (1, H, W) | 0, 1
        dataset_dict["offset_bev"] = offset_bev     # (2, H, W) | -1, 1
        dataset_dict["ego_pose"] = egopose

        #dataset_dict["multi_seg_bev"] = multi_seg_bev       # (num_vehicles, H, W) | 0, 1
        # get first 50 vehicles
        dataset_dict["multi_seg_bev"] = multi_seg_bev[:self.num_queries] 
        dataset_dict["gt_masks"] = seg_bev
        dataset_dict["multi_valid_bev"] = multi_valid_bev[:self.num_queries] 
        dataset_dict["gt_valid"] = valid_bev

        dataset_dict["translation_rotation_list"] = translation_rotation_list[:self.num_queries] 

        if self.get_sem_masks:
            sem_masks = self.load_sem_masks(idx)
            img = torch.from_numpy(sem_masks).float().unsqueeze(1)      # (NCAMS, 1, H, W) | 0, 1
            dataset_dict["images"] = img.repeat(1, 3, 1, 1)             # (NCAMS, 3, H, W) | 0, 1

        return dataset_dict

    def load_sem_masks(self, idx):

        folder = 'train' if self.is_train else 'val'
        masks = np.load(f'/kuacc/users/mbarin22/hpc_run/mask2former4bev/sam_masks/{folder}/{idx}.npz')['masks']  # C, H, W

        return masks

    def choose_cams(self):
        if self.is_train and self.ncams < len(self.cams):
            cams = np.random.choice(self.cams, 
                                    self.ncams,
                                    replace=False)
        else:
            cams = self.cams
        return cams


    def choose_ref_cam(self):

        if self.is_train and self.do_shuffle_cams:
            # randomly sample the ref cam
            refcam_id = np.random.randint(1, self.ncams)# len(self.cams))
        else:
            refcam_id = self.refcam_id

        return refcam_id

    def sample_augmentation(self):
        fH, fW = self.final_dim
        if self.is_train:

            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(fW*resize), int(fH*resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH)/2)
            crop_w = int((newW - fW)/2)

            crop_offset = self.crop_offset
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        else: # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop


    def get_image_data(self, sample, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            samp = self.nusc.get('sample_data', sample['data'][cam])

            imgname = os.path.join(self.dataroot, samp['filename'])
            img = Image.open(imgname)
            W, H = img.size

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            resize_dims, crop = self.sample_augmentation()

            sx = resize_dims[0]/float(W)
            sy = resize_dims[1]/float(H)

            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = img_transform(img, resize_dims, crop)
            imgs.append(totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

            
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),torch.stack(intrins))

    
    def get_binimg(self, rec):

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        # TODO: here we can define img as multi-channel
        # EACH VEHICLE GOES TO SEPARATE CHANNEL
        img_channels = []
        for ii, tok in enumerate(rec['anns']):
            inst = self.nusc.get('sample_annotation', tok)
            
            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if discard_invisible and int(inst['visibility_token']) == 1:
                # filter invisible vehicles
                continue

            channel = np.zeros((self.nx[0], self.nx[1]))
                
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], ii+1.0)
            cv2.fillPoly(channel, [pts], 1.0)
            
            img_channels.append(channel)

        if img_channels == []:
            img_channels = np.zeros((self.nx[0], self.nx[1]))
        else:
            img_channels = np.stack(img_channels)


        return torch.Tensor(img).unsqueeze(0), torch.Tensor(convert_egopose_to_matrix_numpy(egopose)), torch.Tensor(img_channels)

    
    def get_lidar_data(self, rec, nsweeps):

        pts = get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2, dataroot=self.dataroot)
        return pts

    def get_radar_data(self, rec, nsweeps):
        
        pts = get_radar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2, use_radar_filters=self.use_radar_filters, dataroot=self.dataroot)
        
        return torch.Tensor(pts)

    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if int(inst['visibility_token']) == 1:
                vislist.append(torch.tensor(0.0)) # invisible
            else:
                vislist.append(torch.tensor(1.0)) # visible

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            tidlist.append(inst['instance_token'])

            r = box.rotation_matrix
            t = box.center
            l = box.wlh
            l = np.stack([l[1],l[0],l[2]])
            lrt = py.merge_lrt(l, py.merge_rt(r,t))
            lrt = torch.Tensor(lrt)
            lrtlist.append(lrt)
            ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
            # print('rx, ry, rz', rx, ry, rz)
            rs = np.stack([ry*0, ry, ry*0])
            box_ = torch.from_numpy(np.stack([t,l,rs])).reshape(9)
            # print('box_', box_)
            boxlist.append(box_)

        if len(lrtlist):
            lrtlist = torch.stack(lrtlist, dim=0)
            boxlist = torch.stack(boxlist, dim=0)
            vislist = torch.stack(vislist, dim=0)
            # tidlist = torch.stack(tidlist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros((0))
            # tidlist = torch.zeros((0))
            tidlist = []

        return lrtlist, boxlist, vislist, tidlist

    
    def get_seg_bev(self, lrtlist_cam, vislist):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        seg = np.zeros((self.Z, self.X))
        val = np.ones((self.Z, self.X))

        corners_cam = geom.get_xyzlist_from_lrtlist(lrtlist_cam) # B, N, 8, 3
        y_cam = corners_cam[:,:,:,1] # y part; B, N, 8
        corners_mem = self.vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), self.Z, self.Y, self.X).reshape(B, N, 8, 3)

        # take the xz part
        corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3) # B, N, 8, 2
        # corners_mem = corners_mem[:,:,:4] # take the bottom four

        vehicles = []
        invis_vehicles = []
        xz_centers = []
        num_invis = 0
        for n in range(N):
            _, inds = torch.topk(y_cam[0,n], 4, largest=False)
            pts = corners_mem[0,n,inds].numpy().astype(np.int32) # 4, 2
            

            vehicle = np.zeros((self.Z, self.X))
            invis_vehicle = np.ones((self.Z, self.X)) 

            # if this messes in some later conditions,
            # the solution is to draw all combos
            pts = np.stack([pts[0],pts[1],pts[3],pts[2]])   # 4, 2
            # find the center of vehicle
            x_center = np.min(pts[:,0]) + (np.max(pts[:,0]) - np.min(pts[:,0]))/2
            z_center = np.min(pts[:,1]) + (np.max(pts[:,1]) - np.min(pts[:,1]))/2
            x_size = np.max(pts[:,0]) - np.min(pts[:,0])
            z_size = np.max(pts[:,1]) - np.min(pts[:,1])
            xz_centers.append(np.array([x_center/200, z_center/200, x_size/200, z_size/200]))

            # pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(seg, [pts], n+1.0)
            cv2.fillPoly(vehicle, [pts], 1.0)
            
            if vislist[n]==0:
                # draw a black rectangle if it's invisible
                cv2.fillPoly(val, [pts], 0.0)
                cv2.fillPoly(invis_vehicle, [pts], 0.0)
                num_invis += 1

            vehicles.append(vehicle)
            invis_vehicles.append(invis_vehicle)
        
        # if num_invis == N:
        #     vehicles = np.zeros((0, self.Z, self.X))
        #     invis_vehicles = np.ones((0, self.Z, self.X))
        #     #xz_centers = np.zeros((0, 2))
        # else:
        vehicles = np.stack(vehicles, axis=0)
        invis_vehicles = np.stack(invis_vehicles, axis=0)
        
        xz_centers = np.stack(xz_centers, axis=0)               # invisible included


        return torch.Tensor(seg).unsqueeze(0), torch.Tensor(val).unsqueeze(0), torch.Tensor(vehicles), torch.Tensor(invis_vehicles), torch.Tensor(xz_centers)  # 1, Z, X


    def get_center_and_offset_bev(self, lrtlist_cam, seg_bev):
        B, N, D = lrtlist_cam.shape
        assert(B==1)

        lrtlist_mem = self.vox_util.apply_mem_T_ref_to_lrtlist(
            lrtlist_cam, self.Z, self.Y, self.X)
        clist_cam = geom.get_clist_from_lrtlist(lrtlist_cam)
        lenlist, rtlist = geom.split_lrtlist(lrtlist_cam) # B,N,3
        rlist_, tlist_ = geom.split_rt(rtlist.reshape(B*N, 4, 4))

        x_vec = torch.zeros((B*N, 3), dtype=torch.float32, device=rlist_.device)
        x_vec[:, 0] = 1 # 0,0,1 
        x_rot = torch.matmul(rlist_, x_vec.unsqueeze(2)).squeeze(2)

        rylist = torch.atan2(x_rot[:, 0], x_rot[:, 2]).reshape(N)
        rylist = geom.wrap2pi(rylist + np.pi/2.0)

        radius = 3
        center, offset = self.vox_util.xyz2circles_bev(clist_cam, radius, self.Z, self.Y, self.X, already_mem=False, also_offset=True)

        masklist = torch.zeros((1, N, 1, self.Z, 1, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            masklist[0,n,0,:,0] = (inst.squeeze() > 0.01).float()

        size_bev = torch.zeros((1, 3, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            size_bev[0,:,inst] = lenlist[0,n].unsqueeze(1)

        ry_bev = torch.zeros((1, 1, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            ry_bev[0,:,inst] = rylist[n]
            
        ycoord_bev = torch.zeros((1, 1, self.Z, self.X), dtype=torch.float32)
        for n in range(N):
            inst = (seg_bev==(n+1)).float() # 1, Z, X
            inst = inst.reshape(self.Z, self.X) > 0.01
            ycoord_bev[0,:,inst] = tlist_[n,1] # y part

        offset = offset * masklist
        offset = torch.sum(offset, dim=1) # B,3,Z,Y,X

        min_offset = torch.min(offset, dim=3)[0] # B,2,Z,X
        max_offset = torch.max(offset, dim=3)[0] # B,2,Z,X
        offset = min_offset + max_offset
        
        center = torch.max(center, dim=1, keepdim=True)[0] # B,1,Z,Y,X
        center = torch.max(center, dim=3)[0] # max along Y; 1,Z,X
        
        return center.squeeze(0), offset.squeeze(0), size_bev.squeeze(0), ry_bev.squeeze(0), ycoord_bev.squeeze(0) # 1,Z,X; 2,Z,X; 3,Z,X; 1,Z,X






class NuScenesDatasetWrapper:
    '''
        Initialize training and validation datasets.
    '''
    def __init__(self, args):
        self.args = args

        print('Loading NuScenes version', args.version, 'from', args.dataset_path)

        self.nusc = NuScenes(
                    version='v1.0-{}'.format(args.version),
                    dataroot=args.dataset_path, #os.path.join(self.dataroot, 'v1.0-{}'.format(self.version)), # self.dataroot, #
                    verbose=False
                    )
        print('Done loading NuScenes version', args.version)
    
    def train(self):
        self.args.is_train = True
        return NuscenesDataset(self.args, self.nusc)
    
    def val(self):
        self.args.is_train = False
        return NuscenesDataset(self.args, self.nusc)
        