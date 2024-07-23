import h5py
import torch
import pandas as pd
import sys 
import os

file1 = sys.argv[1]
file2 = sys.argv[2]


feats_stroma, coord_stroma = [],[]
with h5py.File(file1, mode="r") as f:
        coord_stroma.append(torch.from_numpy(f["coords"][:]).float())
        feats_stroma.append(torch.from_numpy(f["feats"][:]).float())

feats_tumor, coord_tumor = [],[]
with h5py.File(file2, mode="r") as f:
        coord_tumor.append(torch.from_numpy(f["coords"][:]).float())
        feats_tumor.append(torch.from_numpy(f["feats"][:]).float())


coord_stroma_df=pd.DataFrame(torch.cat(coord_stroma).numpy(),columns=['z','x','y','pred1','pred2'])
coord_tumor_df=pd.DataFrame(torch.cat(coord_tumor).numpy(),columns=['z','x','y','pred1','pred2'])

concatenated_coord = pd.concat([coord_stroma_df, coord_tumor_df], axis=0)
concatenated_features = [torch.cat([feat1, feat2], dim=0) for feat1, feat2 in zip(feats_stroma, feats_tumor)]

sorted_coord = concatenated_coord.sort_values(by=['x', 'y'])
sorted_indices = sorted_coord.index
sorted_features_tensor = torch.index_select(concatenated_features[0], dim=0, index=torch.tensor(sorted_indices))

slidename= os.path.basename(file1)[0:-10]
out_dir = os.path.dirname(os.path.abspath(file1))
sorted_coord_tensor = [torch.tensor(sorted_coord.values)]


with h5py.File(f'{out_dir}/{slidename}_cTransPath.h5', 'w') as f:
    f['coords'] = sorted_coord_tensor
    f['feats'] = sorted_features_tensor
    f['type'] = 'all_neo_sort'
    f['extractor'] = 'cTransPath'