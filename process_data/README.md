# step1 

Go to the link 
https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive 
to download data from 2017-2024.

# step2

Filter the file into daily format

python step0_filter_feature.py

# step3

get all stations in a csv file

python step1_get_union_station.py

# step4 (have finished)

compute the mean and std of data

python step2_climatology.py

# step5

generate data for dgl.

python step3_generate_graph_dgl_step.py

# step6 (have finished)
generate sh embedding data

python step4_process_sh_embedding.py

# step7 (have finished)
generate percentile data for sedi calculation 

python step4_process_sh_embedding.py