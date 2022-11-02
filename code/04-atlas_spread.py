# %%
from os.path import join as ospj
import pickle

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
data_path = "../data/"
# %%
with open(ospj(data_path, "electrode_localization", 'elec_ROI.pkl'), 'rb') as handle:
    elec_ROI = pickle.load(handle)
atlas_info = pd.read_csv(ospj(data_path, "electrode_localization", 'dkt_atlas_info.csv'))
seizure_table = pd.read_excel("../data/metadata/seizure_table.xlsx")
patient_cohort = pd.read_excel("../data/metadata/patient_cohort_thresholds.xlsx")

def region_list_to_vol(region_list, atlas_info=atlas_info):
    '''
    Takes in a column of lists of region IDs and return total volume and number of regions
    '''
    # if isinstance(region_list, list):
    #     all_regions = []
    #     for i in region_list:
    #         all_regions.extend(i)
    #     all_regions = list(set(all_regions))
    # else:
    #     all_regions = [region_list]
    # all_regions_info = atlas_info[atlas_info['regionID'].isin(all_regions)]

    all_regions_info = atlas_info[atlas_info['regionID'].isin(region_list)]
    n_regions = len(all_regions_info)
    total_vol = all_regions_info['volume (mm^3)'].sum()
    return n_regions, total_vol

# %%
for index, row in seizure_table.iterrows():
    pt = row['Patient']
    sz_num = row['Seizure number']
    threshold = patient_cohort[patient_cohort['Patient'] == pt]['threshold'].values[0]
    pt_data_path = ospj(data_path, pt)

    # recruited = pd.read_csv(ospj(pt_data_path, f"recruited_schindler_shatthresh-ptspecific_winthresh-{threshold}_sz-{sz_num}.csv"))['0']
    recruited = np.load(ospj(pt_data_path, f"recruited_schindler_sz-{sz_num}.npy"))
    pt_elec_ROI = elec_ROI[pt]
    recruited_elecs = pt_elec_ROI.iloc[list(recruited), :]

    n_regions, total_vol = region_list_to_vol(recruited_elecs['Max Region'].values)

    seizure_table.at[index, 'Number of Regions'] = int(n_regions)
    seizure_table.at[index, 'Total Volume (cm^3)'] = total_vol / 1000

seizure_table.to_excel(ospj(data_path, "metadata", 'seizure_metadata_with_atlas_spread.xlsx'))

# # %%
# import matplotlib.pyplot as plt

# seizure_metadata_me = pd.read_excel(ospj(data_path, "seizure_metadata_with_atlas_spread.xlsx"))
# seizure_metadata_andy = pd.read_excel(ospj(data_path, "sz_metadata_with_spread_andy.xlsx"))

# for name, seizure_metadata in zip(["Absolute Slope", "Wavenet"], [seizure_metadata_me, seizure_metadata_andy]):
#     fig, ax = plt.subplots()
#     seizure_metadata.boxplot(column='Total Volume (cm^3)', by='Seizure category', ax=ax, grid=False, showfliers=False)
#     fig.suptitle(name)
#     fig.show()
    
#     fig, ax = plt.subplots()
#     seizure_metadata.boxplot(column='Number of Regions', by='Seizure category', ax=ax, grid=False, showfliers=False)
#     fig.suptitle(name)
#     fig.show()

# # %%
# display(seizure_metadata[seizure_metadata['Seizure category'] == 'Focal'].sort_values('Total Volume (cm^3)', ascending=False))
# # %%
# display(seizure_metadata[seizure_metadata['Seizure category'] == 'FBTCS'].sort_values('Total Volume (cm^3)', ascending=True))

# %%
# %%
