import pandas as pd
import glob
import os

def merge_csvs(folder_path):
    file_pattern = os.path.join(folder_path, "*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("No csv file found.")
        return

    print(f"Target files: {all_files}")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)

    final_df = combined_df.groupby('QIDX', as_index=False).first()

    return final_df

if __name__ == "__main__":
    target_folder = "output/"

    final_df = merge_csvs(target_folder)

    if final_df is not None:
        final_df.to_csv("output/result_rag.csv", index=False, encoding='utf-8-sig')