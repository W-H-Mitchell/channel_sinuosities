import pandas as pd

# get the import into the right format
def clean(import_filename):
    df = pd.read_csv(import_filename, encoding='utf-8')
    df.drop_duplicates(keep="first", inplace=True) # remove duplicates from centreline
    df.columns = ['centreline']
    df = df.astype(str)
    delete_items = ['(', ')']
    for item in delete_items:
        df["centreline"] = df['centreline'].str.replace(item, '')
    split_df = pd.DataFrame(df.centreline.str.split(", ").tolist(), columns=["x", "y", "z"]) # split the dataframe into x and y
    split_df["y"] = split_df["y"].str.replace(",","") # remove the commas
    del split_df['z']
    # Change export filename below
    split_df.to_csv(r'/Users/whamitchell/Documents/python/channel_sinuosities/ChSev_centreline.csv', index = False) 

clean("Ch1_xy_points.csv")