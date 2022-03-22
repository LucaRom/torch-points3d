import os
import numpy as np
import logging
import torch
import laspy
import random
import os.path as osp
import omegaconf

from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.core.data_transform.transforms import RandomSphere, GridCylinderSampling, GridSphereSampling, RandomCylinder
from torch_points3d.core.data_transform.grid_transform import GridSampling3D
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

log = logging.getLogger(__name__)

################################### Datasets general notes and structure ###################################

"""
    The dataset is devided in 3 main classes : 
        - stjohns
        - stjohnsSampling
        - stjohnsWrapper
        
    - stjonhs is the main class. It takes the raw data and processes it into .pt files. It is usually where you would
    change code.
    
    - stjohnsSampling is built on stjohns class. It is used as a way to sample the data created with the main class, 
    mainly to handle the number of samples per epoch and per split. Only the __len__ and get() method are overwritten.
    
    - stjohnsWrapper is a final wrapper that creates the different dataset (train, val test) as needed in the framework.
    It calls stjohnsSampling with the desired arguments for the parameters. It is this function that is called in the 
    config files for a training. (data=segmentation/stjonhs)

    NOTES : 
"""

################################### Config files integration ###################################
# Calling overrided conf .yaml file to use extra parameters
"""
    This is a quick workaround to pass arguments/parameters from the config file
    
    NOTES : The config file called is the overrided one from the OUTPUT folder of the current training/eval/etc run
            This way you can call parameters from all conf files since they are grouped in the output conf, but more
            importantly, it allows to use overrided parameters passed to a specific job (ie, from the command line).
            THEREFORE following lines and called parameters/arguments will raise errors if called outside of a run.
                    
            This was first created to allow using different config files when running simultaneous run from the command
            line (ie. on 2 different GPUs). 
"""

output_dir = os.getcwd()
config_path = os.path.join(output_dir, ".hydra/config.yaml")
config_cfg = omegaconf.OmegaConf.load(config_path)

################################### Paths and datasets ###################################

# Settings paths to raw dataset and splitting into train/val/test
DIR = os.path.dirname(os.path.realpath(__file__))
dataroot = os.path.join(DIR + "/../../../data/stjohns/raw")
#dataroot = os.path.join(DIR + "/../../../../torch-points3d.old/data/stjohns/raw") #to old folder
print(dataroot)

# Full raw dataset
train_list_full = [f for f in os.listdir(os.path.join(dataroot, "train")) if f.endswith('.las')]
val_list_full = [f for f in os.listdir(os.path.join(dataroot, "val")) if f.endswith('.las')]
test_list_full = [f for f in os.listdir(os.path.join(dataroot, "test")) if f.endswith('.las')]

# Small dataset for debug/tests
train_list_small = random.choices(train_list_full, k=2)
val_list_small = random.choices(val_list_full, k=1)
test_list_small = random.choices(test_list_full, k=2)

# Visual dataset for debug/tests or very small tests/debug
train_list_vsmall = random.choices(train_list_full, k=1)
val_list_vsmall = random.choices(val_list_full, k=1)
test_list_vsmall = random.choices(test_list_full, k=1)

# Proper set is chosen according to the config file (raw_folder_set)
# Uses of the small dataset is meant for quicker tests/debugging
if config_cfg.data.raw_folder_set == "full":
    train_list = train_list_full
    train_spp = 10000
    val_list = val_list_full
    val_spp = 2000
    test_list = test_list_full
    test_spp = -1
    las_list = train_list_full + val_list_full + test_list_full
elif config_cfg.data.raw_folder_set == "small":
    train_list = train_list_small
    train_spp = 88
    val_list = val_list_small
    val_spp = 88
    test_list = test_list_small
    test_spp = -1
    las_list = train_list_small + val_list_small + test_list_small
else:
    train_list = train_list_vsmall
    train_spp = 88
    val_list = val_list_vsmall
    val_spp = 88
    test_list = test_list_vsmall
    test_spp = -1
    las_list = train_list_vsmall + val_list_vsmall + test_list_vsmall

################################### Labels ###################################
# Those variables are used while remapping labels and when the number of class
# is needed

NUM_CLASSES = 7

CLASS_LABELS = (
    "unclassified",
    "ground",
    "building",
    "noise",
    "water",
    "bridge deck",
    "reserved"
)

VALID_CLASS_IDS = [1, 2, 6, 7, 9, 17, 18]

################################### utils ###################################
'''
 gss_sampler :   GridSphereSampling fit the point clouds to a grid and samples a sphere around the center point.
 _grid_sampler : GridSampling3d resamples the dataset with the center point of a voxel of the set size. This is used to
                 reduce the number of points.
 rs_sampler :    RandomSphereSampling samples a random sphere of points within the dataset, it is slightly biased to 
                 favor under represented classes.     
 rc_sampler :    RandomCylinderSampling same as rs_sampler but with a cylinder shape/               
'''

# Samplers arguments from config file
#gss_radius = config_cfg.data.radius_param       # GridSphereSampling radius
#gss_grid = config_cfg.data.grid_param           # GridSphereSampling grid size
gss_radius = 10
gss_grid = 10

rsc_radius = config_cfg.data.sampler_radius     # RandomCylinder and RandomSphere Sampling radius
gs3d_grid = config_cfg.data.first_subsampling   # GridSphere3d grid size (minimum distance between points)


# Samplers used in dataset creation
gss_sampler = GridSphereSampling(radius=gss_radius, grid_size=gss_grid, delattr_kd_tree=True, center=False)
rs_sampler = RandomSphere(radius=rsc_radius, strategy="freq_class_based")
rc_sampler = RandomCylinder(radius=rsc_radius, strategy="freq_class_based")
_grid_sampler = GridSampling3D(size=gs3d_grid)

################################### Memory dataset Main Class ###################################

class stjohns(InMemoryDataset):
    """
    Class to handle stjohns dataset for segmentation task.
    Most of the methods are inherited from the InMemoryDataset (in_memory_dataset.py) which also inherit methods
    from Dataset (dataset.py) wich is based on pytorch dataset class.

    NOTES : - If a method is called, it's either an existing one that is being override (from the parent class) or a
            new one created for the dataset.
            - Errors may arise from methods called by default in parent classes.
    """

    CLASS_LABELS = CLASS_LABELS
    VALID_CLASS_IDS = VALID_CLASS_IDS

    def __init__(self, root, split=None, radius=None, grid_size_d=None, transform=None, pre_transform=None, pre_filter=None):

        # Variable are set in the class from the arguments so the can be called in the class' methods
        self._split = split
        self._radius = radius
        self._grid_size_d = grid_size_d
        self._first_subsampling = config_cfg.data.first_subsampling
        self._main_sampler = config_cfg.data.main_sampler

        self.valid_class_idx = [idx for idx in self.VALID_CLASS_IDS]

        # log infos are printed in the output log file of each run
        log.info(f"Actual split is {self._split}")
        log.info(f"Sampler's parameters : Radius = {self._radius}, Grid = {self._grid_size_d}")
        log.info(f"Parameter first_subsampling is set to : {self._first_subsampling}")
        log.info(f"Number of files : {len(las_list)} using file from {config_cfg.data.processed_folder_name}")

        # super() specifies methods inherited from the superclass (here "InMemoryDataset")
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        # This loads existing processed files (it runs after the process method when they do not exist yet)
        # processed_paths are defined in the "processed_file_names(self)" method below
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError("Split %s not recognised" % split)

        # Calls the _load_data method to load selected path
        self._load_data(path)

    @property
    def processed_dir(self):
        """
        Set the name of the processed files folder
        Here it is fetched from the config parameter "processed_folder_name" in stjohns.yaml
        """
        return osp.join(self.root, config_cfg.data.processed_folder_name)
        #return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        """
        This method returns a list of file names for the raw files needed for the dataset. If file names are missing, it
        will trigger the download method. Since the dataset is managed manually outside the class, the method is passed.

        NOTES : This method is needed for the dataset to work, that's why 'pass' is used.
        """
        return las_list

    @property
    def processed_file_names(self):
        """
        This method defines the name of the processed datasets that are processed and grouped in a .pt file. Those
        files names also define the processed_paths list (starting from 0).
        """
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        """
        This method would be trigger by an incomplete list of raw files (raw_file_names). Since the raw files are
        managed manually, this method is passed.

        NOTES : This method is needed for the dataset to work, that's why 'pass' is used.
        """
        pass

    def process(self):
        """
        This method gives the instructions to handle the raw data and process it if they are not already according to
        the specific dataset (_split). Each .las file is processed and group to respective dataset in the .pt format
        (train.pt, val.pt, test.pt.old).

        It calls the split_process method to process data.

        Information is extracted from .las file to feed the Data(pos, x, y) class where :
            pos = x, y, z (coordinates)
            x = list of features (if any)
            y = labels (if any)

        NOTES : - The x and y arguments from the Data() class are not related to the x and y coordinates needed for
        pos.
        """

        # Raw list are set at the beginning in "Paths and datasets" section
        # Details on pp_paths are found in the "processed_file_names" method
        if self._split == "train":
            pp_paths = self.processed_paths[0]
            current_split = "train"
            raw_list = train_list
        elif self._split == "val":
            pp_paths = self.processed_paths[1]
            current_split = "val"
            raw_list = val_list
        elif self._split == "test":
            pp_paths = self.processed_paths[2]
            current_split = "test"
            raw_list = test_list
        else:
            raise ValueError("Split %s not recognised" % self._split)

        self.split_process(current_split=current_split, raw_list=raw_list, pp_paths=pp_paths)

    @property
    def num_classes(self):
        return len(VALID_CLASS_IDS)

    def _remap_labels(self, semantic_label):
        """Remaps labels to [0 ; num_labels -1]. """
        new_labels = semantic_label.clone()
        mapping_dict = {indice: idx for idx, indice in enumerate(self.valid_class_idx)}
        for source, target in mapping_dict.items():
            mask = semantic_label == source
            new_labels[mask] = target

        return new_labels

    def split_process(self, current_split, raw_list, pp_paths):
        """
        Method called to process raw data according to the actual split/dataset. It uses laspy as the main librairy to
        extract the data.

        Information is extracted from .las file to feed the Data(pos, x, y) class where :
            pos = x, y, z (coordinates)
            x = list of features (if any)
            y = labels (if any)

        NOTES : - The x and y arguments from the Data() class are not related to the x and y coordinates needed for
        pos.

        :param current_split: Specifies which dataset is actually processed (train, val or test)
        :param raw_list: Path to the raw data folder
        :param pp_paths: Processed path (see processed_file_names())
        """

        # Check if the processed file already exist for this split, if not, proceed with the processing
        if os.path.exists(pp_paths):
            print(f"Processed file for {current_split} already exists, skipping processing")
        else:
            if self._main_sampler == "sphere":  # Select sampler according to config file
                main_sampler = rs_sampler
            else:
                main_sampler = rc_sampler

            data_list = []
            for j, i in enumerate(raw_list, 1):
                # Load the .las file and extract needed data (using laspy)
                las_file = laspy.read(os.path.join(dataroot, current_split, i))
                las_xyz = np.stack([las_file.x, las_file.y, las_file.z], axis=1)
                las_label = np.array(las_file.classification).astype(np.int)
                y = torch.from_numpy(las_label)
                y = self._remap_labels(y) # Remapping label necessary is not [0, n] already

                # Feed extracted data to the Data() class. Data is also resampled for train and val.
                data = Data(pos=torch.from_numpy(las_xyz).type(torch.float), y=y)

                # This current strategy creates a pool of 30 000 random samples that will be called during the
                # training process. num_needed defines the number of samples required for each tile
                if current_split == "train":
                    #num_needed = 30000 // len(raw_list)
                    #num_needed = 10 // len(raw_list)
                    #num_needed = 1000
                    #self._sort_samples(num_needed, main_sampler, j, data_list, data)

                    data_samples = gss_sampler(data.clone())  # gss = GridSphereSampling

                    # If sampler results in a list, we first remove samples with no points, else we save the whole data
                    if isinstance(data_samples, list):
                        print(f"Filtering {len(data_samples)} data samples")
                        for my_sample in data_samples:
                            if len(my_sample.y) > 1:
                                data_list.append(my_sample)
                    else:
                        data_list.append(data_samples)


                    log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])

                elif current_split == "val":
                    #num_needed = 10 // len(raw_list)
                    #num_needed = 300
                    #self._sort_samples(num_needed, main_sampler, j, data_list, data)

                    data_samples = gss_sampler(data.clone())  # gss = GridSphereSampling

                    # If sampler results in a list, we first remove samples with no points, else we save the whole data
                    if isinstance(data_samples, list):
                        print(f"Filtering {len(data_samples)} data samples")
                        for my_sample in data_samples:
                            if len(my_sample.y) > 1:
                                data_list.append(my_sample)
                    else:
                        data_list.append(data_samples)


                    log.info("Processed file %s, nb points = %i", i, data.pos.shape[0])

                # Test data is handled differently because we want to test over all the points
                elif current_split == "test":
                    data_samples = gss_sampler(data.clone())  # gss = GridSphereSampling

                    # If sampler results in a list, we first remove samples with no points, else we save the whole data
                    if isinstance(data_samples, list):
                        print(f"Filtering {len(data_samples)} data samples")
                        for my_sample in data_samples:
                            if len(my_sample.y) > 1:
                                data_list.append(my_sample)
                    else:
                        data_list.append(data_samples)

                    log.info("Processed file %s, nb points = %i, nb samples = %i", i, data.pos.shape[0],
                             len(data_samples))
                else:
                    raise ValueError("Something is wrong in the split_process method")

            print(f"Saving full data list in {pp_paths}")
            self._save_data(data_list, pp_paths)

    @staticmethod
    def _sort_samples(num_needed, main_sampler, num_file, data_list, data):
        # Quick method to create the random samples and remove data with 1 or 0 points. 0 points data will rise errors
        # while 1 would not, but we still remove it in that case.
        samples_pbar = tqdm(total=num_needed)
        while len(data_list) < (num_file * num_needed):
            random_data = main_sampler(data)
            if len(random_data.y) > 1:
                data_list.append(random_data)
                samples_pbar.update(1)
            else:
                continue
        samples_pbar.close()

    def _save_data(self, data_list, pp_path):
        data, slices = self.collate(data_list)
        torch.save((data, slices), pp_path)

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)

class stjohnsSampling(stjohns):
    """
    Class that allows the sampling of the stjohns dataset, It uses the stjohns class as the parent
    class to which we add the __len__ and get() method to handle the dataset.

    The __len__ method specifies then number of samples needed
    The get() method implements a random sampling strategy that favors underreprented labels

        :param root:
        :param sample_per_epoch:
        :param radius: Radius of the sphere that samples the dataset
        :param grid_size_d: Placeholder if needed for another sampling method
        :param split: Specifies which dataset is actually processed (train, val or test)
        :param transform: Carries transform called in the config file
        :param pre_transform: Carries pre_transform called in the config file
        :param pre_filter: Carries pre_filter called in the config file
    """

    # Actually, radius and grid_size_d are not used for the moment in this class
    def __init__(self, root, sample_per_epoch=None, radius=None, grid_size_d=None, split=None, transform=None,
                 pre_transform=None, pre_filter=None):

        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid = grid_size_d

        super().__init__(root, split=split, radius=radius, grid_size_d=grid_size_d, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter)

    def __len__(self):  # Defines length of dataset (number of samples fetch per epoch)
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._datas)

    def get(self, idx):  # Creates the index, based on __len__, to feed the dataloader (in this case)
        if self._split == "test":
            return self._datas[idx].clone()
        else:  # Random idx is created to sample data from full length dataset instead of sample_per_epoch
            random_idx = random.randint(0, len(self._datas)-1)
            return self._datas[random_idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _save_data(self, data_list, pp_path):
        torch.save((data_list), pp_path)

    def _load_data(self, path):
        self._datas = torch.load(path)

class stjohnsWrapper(BaseDataset):
    """ Wrapper around stjohnsSamplins that creates train,val and test datasets.
    This class is the one that should be called in config files.

    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)


        self.train_dataset = stjohns(
            self._data_path,
            split="train",
            #sample_per_epoch=train_spp,  # -1 for all
            radius=config_cfg.data.radius_param,
            grid_size_d=config_cfg.data.grid_param,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )

        self.val_dataset = stjohns(
            self._data_path,
            split="val",
            #sample_per_epoch=val_spp,  # -1 for all
            radius=config_cfg.data.radius_param,
            grid_size_d=config_cfg.data.grid_param,
            transform=self.val_transform,
            pre_transform=self.pre_transform,
        )
        
        self.test_dataset = stjohns(
           self._data_path,
           split="test",
           radius=config_cfg.data.radius_param,
           #sample_per_epoch=test_spp,  # -1 for all
           grid_size_d=config_cfg.data.grid_param,
           transform=self.test_transform,
           pre_transform=self.pre_transform,
       )


        # self.train_dataset = stjohnsSampling(
        #     self._data_path,
        #     split="train",
        #     sample_per_epoch=train_spp,  # -1 for all
        #     radius=config_cfg.data.radius_param,
        #     grid_size_d=config_cfg.data.grid_param,
        #     transform=self.train_transform,
        #     pre_transform=self.pre_transform,
        # )
        #
        # self.val_dataset = stjohnsSampling(
        #     self._data_path,
        #     split="val",
        #     sample_per_epoch=val_spp,  # -1 for all
        #     radius=config_cfg.data.radius_param,
        #     grid_size_d=config_cfg.data.grid_param,
        #     transform=self.val_transform,
        #     pre_transform=self.pre_transform,
        # )
        # self.test_dataset = stjohns(
        #     self._data_path,
        #     split="test",
        #     radius=config_cfg.data.radius_param,
        #     grid_size_d=config_cfg.data.grid_param,
        #     transform=self.test_transform,
        #     pre_transform=self.pre_transform,
        # )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
