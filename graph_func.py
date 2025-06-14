#
# import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
import os
import torchvision.models as models
import torchvision.transforms as transforms
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from sklearn.decomposition import PCA 
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import math
import anndata

from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
import cv2

import torch.nn

from efficientnet_pytorch import EfficientNet

    
def image_crop(
        adata,
        save_path,
        library_id=None,
        crop_size=50,
        target_size=224,
        verbose=False,
        ):
    if library_id is None:
       library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
            adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.LANCZOS) ##### 
            tile.resize((target_size, target_size)) ###### 
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names
    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path'] !")
    return adata


    
def read_10X_Visium(path, 
                    genome = None,
                    count_file ='filtered_feature_bc_matrix.h5', 
                    library_id = None, 
                    load_images =True, 
                    quality ='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome = genome,
                        count_file = count_file,
                        library_id = library_id,
                        load_images = load_images,
                        )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path,0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    imagecol=adata.obs['imagecol'].values
    imagerow=adata.obs['imagerow'].values
    return adata


def read_SlideSeq(path, 
                 library_id = None,
                 scale = None,
                 quality = "hires",
                 spot_diameter_fullres= 50,
                 background_color = "white",):

    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))

    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    if scale == None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["x", "y"]].values

    return adata


def read_merfish(path, 
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=50,
                background_color="white",):

    counts = sc.read_csv(os.path.join(path, 'counts.csv')).transpose()
    locations = pd.read_excel(os.path.join(path, 'spatial.xlsx'), index_col=0)
    if locations.min().min() < 0:
        locations = locations + np.abs(locations.min().min()) + 100
    adata = counts[locations.index, :]
    adata.obsm["spatial"] = locations.to_numpy()

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "MERSEQ"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def read_seqfish(path,
                library_id= None,
                scale= 1.0,
                quality= "hires",
                field = 0,
                spot_diameter_fullres = 50,
                background_color = "white",):

    count = pd.read_table(os.path.join(path, 'counts.matrix'), header=None)
    spatial = pd.read_table(os.path.join(path, 'spatial.csv'), index_col=False)

    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)
    spatial = spatial[spatial["Field_of_View"] == field]

    # cells = set(count[''])
    # obs = pd.DataFrame(index=cells)
    adata = AnnData(count)

    if scale == None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "SeqFish"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata

def read_stereoSeq(path,
                bin_size=100,
                is_sparse=True,
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=1,
                background_color="white",
                ):
    from scipy import sparse
    path='D:/data/Stereo-seqMOB/Dataset1_LiuLongQi_MouseOlfactoryBulb/RNA_counts.tsv'
    count = pd.read_csv(path, sep='\t')
    count.dropna(inplace=True)
    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)
    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)
    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]
    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
                 sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()
    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)
    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    adata.obsm['spatial'] = pos

    if scale == None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    return adata


def refine(
    sample_id, 
    pred, 
    dis, 
    shape="hexagon"
    ):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred

class run():
	def __init__(
		self,
		sample_name,
		save_path="./",
		task = "Identify_Domain",
		pre_epochs=1000, 
		epochs=500,
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		):
		self.sample_name = sample_name
		self.save_path = save_path
		self.pre_epochs = pre_epochs
		self.epochs = epochs
		self.task = task
		self.device=device

	def _get_adata(
		self,
		platform,
		data_path,
		sample_name,
		verbose = True,
		):
		assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
		if platform in ['Visium', 'ST']:
			if platform == 'Visium':
				adata = read_10X_Visium(os.path.join(data_path, sample_name))
		elif platform == 'MERFISH':
 			adata = read_merfish(os.path.join(data_path, sample_name))
		elif platform == 'slideSeq':
 			adata = read_SlideSeq(os.path.join(data_path, sample_name))
		elif platform == 'seqFish':
 			adata = read_seqfish(os.path.join(data_path, sample_name))
		elif platform == 'stereoSeq':
 			adata = read_stereoSeq(os.path.join(data_path, sample_name))
		else:
 			raise ValueError(
               				 f"""\
               				 {self.platform!r} does not support.
 	                				""")
		if verbose:
 			save_data_path = Path(os.path.join(self.save_path, "Data", sample_name))
 			save_data_path.mkdir(parents=True, exist_ok=True)
 			adata.write(os.path.join(save_data_path, f'{sample_name}_raw.h5ad'), compression="gzip")
		return adata


	def _get_image_crop(
		self,
		adata,
		sample_name,
		cnnType = 'ResNet50',
		#cnnType = 'efficientnet-b7',
		pca_n_comps = 200,   
		crop_radius = 25, 
		patch_target_size = 224, 
		):
		save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', sample_name))
		save_path_image_crop.mkdir(parents=True, exist_ok=True)
		adata = image_crop(adata, save_path=save_path_image_crop)
		adata = image_feature(adata, pca_components = pca_n_comps, cnnType = cnnType).extract_image_feat()  
		return adata
    




class image_feature:
    def __init__(
        self,
        adata,
        pca_components=200,
        cnnType='ResNet50',
        #cnnType='efficientnet-b7',
        verbose=False,
        #seeds=42,
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        #self.seeds = seeds
        self.cnnType = cnnType

    def load_cnn_model(
        self,
        ):

        if self.cnnType == 'ResNet50':
            cnn_pretrained_model = models.resnet18(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Resnet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
            cnn_pretrained_model.to(self.device)            
        elif self.cnnType == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'DenseNet121':
            cnn_pretrained_model = models.densenet121(pretrained=True)
            cnn_pretrained_model.to(self.device)
        elif self.cnnType == 'Inception_v3':
            cnn_pretrained_model = models.inception_v3(pretrained=True)
            cnn_pretrained_model.to(self.device)
        else:
            raise ValueError(
                    f"""\
                        {self.cnnType} is not a valid type.
                        """)
        return cnn_pretrained_model
    
    
    # def efficientNet_model(self):
    #     efficientnet_versions = {
    #         'efficientnet-b0': 'efficientnet-b0',
    #         'efficientnet-b1': 'efficientnet-b1',
    #         'efficientnet-b2': 'efficientnet-b2',
    #         'efficientnet-b3': 'efficientnet-b3',
    #         'efficientnet-b4': 'efficientnet-b4',
    #         'efficientnet-b5': 'efficientnet-b5',
    #         'efficientnet-b6': 'efficientnet-b6',
    #         'efficientnet-b7': 'efficientnet-b7',
    #     }
    #     if self.cnnType in efficientnet_versions:
    #         model_version = efficientnet_versions[self.cnnType]
    #         cnn_pretrained_model = EfficientNet.from_pretrained(model_version)
    #         cnn_pretrained_model.to(self.device)
    #     else:
    #         raise ValueError(f"{self.cnnType} is not a valid EfficientNet type.")
    #     return cnn_pretrained_model


    def extract_image_feat(
        self,
        ):

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std =[0.229, 0.224, 0.225]),
                          transforms.RandomAutocontrast(),
                          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
                          transforms.RandomInvert(),
                          transforms.RandomAdjustSharpness(random.uniform(0, 1)),
                          transforms.RandomSolarize(random.uniform(0, 1)),
                          transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
                          transforms.RandomErasing()
                          ]
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize(mean=[0.54, 0.51, 0.68], 
        #                   std =[0.25, 0.21, 0.16])]
        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.load_cnn_model()
        #model = self.efficientNet_model()

        #model.fc = torch.nn.LeakyReLU(0.1)
        
        model.eval()


        if "slices_path" not in self.adata.obs.keys():
             raise ValueError("Please run the function image_crop first")

        with tqdm(total=len(self.adata),
              desc="Extract image feature",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path)
                spot_slice = spot_slice.resize((224,224))
                spot_slice = np.asarray(spot_slice, dtype="int32")
                spot_slice = spot_slice.astype(np.float32)
                tensor = img_to_tensor(spot_slice)
                tensor = tensor.resize_(1,3,224,224)
                tensor = tensor.to(self.device)
                result = model(Variable(tensor))
                result_npy = result.data.cpu().numpy().ravel()
                feat_df[spot] = result_npy
                feat_df = feat_df.copy()
                pbar.update(1)
        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()
        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat'] !")
        pca = PCA(n_components=self.pca_components)#, random_state=self.seeds
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())
        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca'] !")
        return self.adata
   

##### generate n
def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    # sample_name = list(adata.uns['spatial'].keys())[0]
    # scalefactors = adata.uns['spatial'][sample_name]['scalefactors']
    # adj_mat = dist <= scalefactors['fiducial_diameter_fullres'] * (n+0.2)
    # adj_mat = adj_mat.astype(int)

    # n_neighbors = np.argpartition(dist, n+1, axis=1)[:, :(n+1)]
    # adj_mat = np.zeros((len(adata), len(adata)))
    # for i in range(len(adata)):
    #     adj_mat[i, n_neighbors[i, :]] = 1

    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### normalze graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def mask_generator(adj_label, N=1):
    idx = adj_label.indices()
    cell_num = adj_label.size()[0]

    list_non_neighbor = []
    for i in range(0, cell_num):
        neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
        n_selected = len(neighbor) * N

        # non neighbors
        total_idx = torch.range(0, cell_num-1, dtype=torch.float32)
        non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
        indices = torch.randperm(len(non_neighbor), dtype=torch.float32)
        random_non_neighbor = indices[:n_selected]
        list_non_neighbor.append(random_non_neighbor)

    x = adj_label.indices()[0]
    y = torch.concat(list_non_neighbor)

    indices = torch.stack([x, y])
    indices = torch.concat([adj_label.indices(), indices], axis=1)

    value = torch.concat([adj_label.values(), torch.zeros(len(x), dtype=torch.float32)])
    adj_mask = torch.sparse_coo_tensor(indices, value)

    return adj_mask


def graph_computing(pos, n):
    from scipy.spatial import distance
    list_x = []
    list_y = []
    list_value = []

    for node_idx in range(len(pos)):
        tmp = pos[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, pos, 'euclidean')
        res = distMat.argsort()
        # tmpdist = distMat[0, res[0][1:params.k + 1]]
        for j in np.arange(1, n + 1):
            list_x += [node_idx, res[0][j]]
            list_y += [res[0][j], node_idx]
            list_value += [1, 1]

    adj = sp.csr_matrix((list_value, (list_x, list_y)))
    adj = adj >= 1
    adj = adj.astype(np.float32)
    return adj


def graph_construction(adata, n=6, dmax=150):
    adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
        # adj_m1 = graph_computing(adata.obsm['spatial'], n=n)
    adj_m2 = generate_adj_mat_1(adata, dmax)
    adj_m1 = sp.coo_matrix(adj_m1)
    adj_m2 = sp.coo_matrix(adj_m2)
    

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()
    adj_m2 = adj_m2 - sp.dia_matrix((adj_m2.diagonal()[np.newaxis, :], [0]), shape=adj_m2.shape)
    adj_m2.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_norm_m2 = preprocess_graph(adj_m2)
    adj_m2 = adj_m2 + sp.eye(adj_m2.shape[0])
    # adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())

    adj_m1 = adj_m1.tocoo()
    shape1 = adj_m1.shape
    values1 = adj_m1.data
    indices1 = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices1, values1, shape1)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    
    adj_m2 = adj_m2.tocoo()
    shape2 = adj_m2.shape
    values2 = adj_m2.data
    indices2 = np.stack([adj_m2.row, adj_m2.col])
    adj_label_m2 = torch.sparse_coo_tensor(indices2, values2, shape2)

    norm_m2 = adj_m2.shape[0] * adj_m2.shape[0] / float((adj_m2.shape[0] * adj_m2.shape[0] - adj_m2.sum()) * 2)
    
    norm_m = 0.5*norm_m1 + 0.5*norm_m2



    # # generate random mask
    # adj_mask = mask_generator(adj_label_m1.to_sparse(), N)

    graph_dict = {
        "adj_norm1": adj_norm_m1,
        "adj_norm2": adj_norm_m2,
        "adj_label1": adj_label_m1.coalesce(),
        "adj_label2": adj_label_m2.coalesce(),
        "norm_value1": norm_m1,
        "norm_value2": norm_m2,
        "norm_value": norm_m,
        # "mask": adj_mask
    }

    return graph_dict


def combine_graph_dict(dict_1, dict_2):
    # TODO add adj_org
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    tmp_adj_label = torch.block_diag(dict_1['adj_label'].to_dense(), dict_2['adj_label'].to_dense())
    graph_dict = {
        "adj_norm": tmp_adj_norm.to_sparse(),
        "adj_label": tmp_adj_label.to_sparse(),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
    }
    return graph_dict