def get_string(version, item):
    return f"synth_data/regression/Xy_N=1000000_Epoch_N=1000_Nt=1_p=32/GP_version={version}/fn={item}.csv"

def get_files_to_process(files_to_process_path, version):
    
    if files_to_process_path == "synth_data/regression/Xy_N=1000000_Epoch_N=1000_Nt=1_p=32/GP_version=":
        return [get_string(version, item) for item in range(1000)]
    


files_to_process = {
    "HENSMAN_file_name_stems" : [
        "data/HENSMAN_X_2012_November_1.csv",
        "data/HENSMAN_X_2012_November_2.csv",
        "data/HENSMAN_X_2012_November_3.csv",
        "data/HENSMAN_X_2012_November_4.csv",
        "data/HENSMAN_X_2012_November_5.csv",
        "data/HENSMAN_X_2012_November_6.csv",
        "data/HENSMAN_X_2012_November_7.csv",
        "data/HENSMAN_X_2012_November_8.csv",
        "data/HENSMAN_X_2012_November_9.csv",
        "data/HENSMAN_X_2012_November_10.csv",
        "data/HENSMAN_X_2012_November_11.csv",
        "data/HENSMAN_X_2012_November_12.csv",
        "data/HENSMAN_X_2012_November_13.csv",
        "data/HENSMAN_X_2012_November_14.csv",
        "data/HENSMAN_X_2012_November_15.csv",
        "data/HENSMAN_X_2012_November_16.csv",
        "data/HENSMAN_X_2012_November_17.csv",
        "data/HENSMAN_X_2012_November_18.csv",
        "data/HENSMAN_X_2012_November_19.csv",
        "data/HENSMAN_X_2012_November_20.csv",
        "data/HENSMAN_X_2012_November_21.csv",
        "data/HENSMAN_X_2012_November_22.csv",
        "data/HENSMAN_X_2012_November_23.csv",
        "data/HENSMAN_X_2012_November_24.csv",
        "data/HENSMAN_X_2012_November_25.csv",
        "data/HENSMAN_X_2012_November_26.csv",
        "data/HENSMAN_X_2012_November_27.csv",
        "data/HENSMAN_X_2012_November_28.csv",
        "data/HENSMAN_X_2012_November_29.csv",
        "data/HENSMAN_X_2012_November_30.csv",
        "data/HENSMAN_X_2012_December_1.csv",
        "data/HENSMAN_X_2012_December_2.csv",
        "data/HENSMAN_X_2012_December_3.csv",
        "data/HENSMAN_X_2012_December_4.csv",
        "data/HENSMAN_X_2012_December_5.csv",
        "data/HENSMAN_X_2012_December_6.csv",
        "data/HENSMAN_X_2012_December_7.csv",
        "data/HENSMAN_X_2012_December_8.csv",
        "data/HENSMAN_X_2012_December_9.csv",
        "data/HENSMAN_X_2012_December_10.csv",
        "data/HENSMAN_X_2012_December_11.csv",
        "data/HENSMAN_X_2012_December_12.csv",
        "data/HENSMAN_X_2012_December_13.csv",
        "data/HENSMAN_X_2012_December_14.csv",
        "data/HENSMAN_X_2012_December_15.csv",
        "data/HENSMAN_X_2012_December_16.csv",
        "data/HENSMAN_X_2012_December_17.csv",
        "data/HENSMAN_X_2012_December_18.csv",
        "data/HENSMAN_X_2012_December_19.csv",
        "data/HENSMAN_X_2012_December_20.csv",
        "data/HENSMAN_X_2012_December_21.csv",
        "data/HENSMAN_X_2012_December_22.csv",
        "data/HENSMAN_X_2012_December_23.csv",
        "data/HENSMAN_X_2012_December_24.csv",
        "data/HENSMAN_X_2012_December_25.csv",
        "data/HENSMAN_X_2012_December_26.csv",
        "data/HENSMAN_X_2012_December_27.csv",
        "data/HENSMAN_X_2012_December_28.csv",
        "data/HENSMAN_X_2012_December_29.csv",
        "data/HENSMAN_X_2012_December_30.csv",
        "data/HENSMAN_X_2012_December_31.csv",
        "data/HENSMAN_X_2013_January_1.csv",
        "data/HENSMAN_X_2013_January_2.csv",
        "data/HENSMAN_X_2013_January_3.csv",
        "data/HENSMAN_X_2013_January_4.csv",
        "data/HENSMAN_X_2013_January_5.csv",
        "data/HENSMAN_X_2013_January_6.csv",
        "data/HENSMAN_X_2013_January_7.csv",
        "data/HENSMAN_X_2013_January_8.csv",
        "data/HENSMAN_X_2013_January_9.csv",
        "data/HENSMAN_X_2013_January_10.csv",
        "data/HENSMAN_X_2013_January_11.csv",
        "data/HENSMAN_X_2013_January_12.csv",
        "data/HENSMAN_X_2013_January_13.csv",
        "data/HENSMAN_X_2013_January_14.csv",
        "data/HENSMAN_X_2013_January_15.csv",
        "data/HENSMAN_X_2013_January_16.csv",
        "data/HENSMAN_X_2013_January_17.csv",
        "data/HENSMAN_X_2013_January_18.csv",
        "data/HENSMAN_X_2013_January_19.csv",
        "data/HENSMAN_X_2013_January_20.csv",
        "data/HENSMAN_X_2013_January_21.csv",
        "data/HENSMAN_X_2013_January_22.csv",
        "data/HENSMAN_X_2013_January_23.csv",
        "data/HENSMAN_X_2013_January_24.csv",
        "data/HENSMAN_X_2013_January_25.csv",
        "data/HENSMAN_X_2013_January_26.csv",
        "data/HENSMAN_X_2013_January_27.csv",
        "data/HENSMAN_X_2013_January_28.csv",
        "data/HENSMAN_X_2013_January_29.csv",
        "data/HENSMAN_X_2013_January_30.csv",
        "data/HENSMAN_X_2013_January_31.csv",
        "data/HENSMAN_X_2013_February_1.csv",
        "data/HENSMAN_X_2013_February_2.csv",
        "data/HENSMAN_X_2013_February_3.csv",
        "data/HENSMAN_X_2013_February_4.csv",
        "data/HENSMAN_X_2013_February_5.csv",
        "data/HENSMAN_X_2013_February_6.csv",
        "data/HENSMAN_X_2013_February_7.csv",
        "data/HENSMAN_X_2013_February_8.csv",
        "data/HENSMAN_X_2013_February_9.csv",
        "data/HENSMAN_X_2013_February_10.csv",
        "data/HENSMAN_X_2013_February_11.csv",
        "data/HENSMAN_X_2013_February_12.csv",
        "data/HENSMAN_X_2013_February_13.csv",
        "data/HENSMAN_X_2013_February_14.csv",
        "data/HENSMAN_X_2013_February_15.csv",
        "data/HENSMAN_X_2013_February_16.csv",
        "data/HENSMAN_X_2013_February_17.csv",
        "data/HENSMAN_X_2013_February_18.csv",
        "data/HENSMAN_X_2013_February_19.csv",
        "data/HENSMAN_X_2013_February_20.csv",
        "data/HENSMAN_X_2013_February_21.csv",
        "data/HENSMAN_X_2013_February_22.csv",
        "data/HENSMAN_X_2013_February_23.csv",
        "data/HENSMAN_X_2013_February_24.csv",
        "data/HENSMAN_X_2013_February_25.csv",
        "data/HENSMAN_X_2013_February_26.csv",
        "data/HENSMAN_X_2013_February_27.csv",
        "data/HENSMAN_X_2013_February_28.csv",
        "data/HENSMAN_X_2013_March_1.csv",
        "data/HENSMAN_X_2013_March_2.csv",
        "data/HENSMAN_X_2013_March_3.csv",
        "data/HENSMAN_X_2013_March_4.csv",
        "data/HENSMAN_X_2013_March_5.csv",
        "data/HENSMAN_X_2013_March_6.csv",
        "data/HENSMAN_X_2013_March_7.csv",
        "data/HENSMAN_X_2013_March_8.csv",
        "data/HENSMAN_X_2013_March_9.csv",
        "data/HENSMAN_X_2013_March_10.csv",
        "data/HENSMAN_X_2013_March_11.csv",
        "data/HENSMAN_X_2013_March_12.csv",
        "data/HENSMAN_X_2013_March_13.csv",
        "data/HENSMAN_X_2013_March_14.csv",
        "data/HENSMAN_X_2013_March_15.csv",
        "data/HENSMAN_X_2013_March_16.csv",
        "data/HENSMAN_X_2013_March_17.csv",
        "data/HENSMAN_X_2013_March_18.csv",
        "data/HENSMAN_X_2013_March_19.csv",
        "data/HENSMAN_X_2013_March_20.csv",
        "data/HENSMAN_X_2013_March_21.csv",
        "data/HENSMAN_X_2013_March_22.csv",
        "data/HENSMAN_X_2013_March_23.csv",
        "data/HENSMAN_X_2013_March_24.csv",
        "data/HENSMAN_X_2013_March_25.csv",
        "data/HENSMAN_X_2013_March_26.csv",
        "data/HENSMAN_X_2013_March_27.csv",
        "data/HENSMAN_X_2013_March_28.csv",
        "data/HENSMAN_X_2013_March_29.csv",
        "data/HENSMAN_X_2013_March_30.csv",
        "data/HENSMAN_X_2013_March_31.csv",
        "data/HENSMAN_X_2013_April_1.csv",
        "data/HENSMAN_X_2013_April_2.csv",
        "data/HENSMAN_X_2013_April_3.csv",
        "data/HENSMAN_X_2013_April_4.csv",
        "data/HENSMAN_X_2013_April_5.csv",
        "data/HENSMAN_X_2013_April_6.csv",
        "data/HENSMAN_X_2013_April_7.csv",
        "data/HENSMAN_X_2013_April_8.csv",
        "data/HENSMAN_X_2013_April_9.csv",
        "data/HENSMAN_X_2013_April_10.csv",
        "data/HENSMAN_X_2013_April_11.csv",
        "data/HENSMAN_X_2013_April_12.csv",
        "data/HENSMAN_X_2013_April_13.csv",
        "data/HENSMAN_X_2013_April_14.csv",
        "data/HENSMAN_X_2013_April_15.csv",
        "data/HENSMAN_X_2013_April_16.csv",
        "data/HENSMAN_X_2013_April_17.csv",
        "data/HENSMAN_X_2013_April_18.csv",
        "data/HENSMAN_X_2013_April_19.csv",
        "data/HENSMAN_X_2013_April_20.csv",
        "data/HENSMAN_X_2013_April_21.csv",
        "data/HENSMAN_X_2013_April_22.csv",
        "data/HENSMAN_X_2013_April_23.csv",
        "data/HENSMAN_X_2013_April_24.csv",
        "data/HENSMAN_X_2013_April_25.csv",
        "data/HENSMAN_X_2013_April_26.csv",
        "data/HENSMAN_X_2013_April_27.csv",
        "data/HENSMAN_X_2013_April_28.csv",
        "data/HENSMAN_X_2013_April_29.csv",
        "data/HENSMAN_X_2013_April_30.csv",
        "data/HENSMAN_X_2013_May_1.csv",
        "data/HENSMAN_X_2013_May_2.csv",
        "data/HENSMAN_X_2013_May_3.csv",
        "data/HENSMAN_X_2013_May_4.csv",
        "data/HENSMAN_X_2013_May_5.csv",
        "data/HENSMAN_X_2013_May_6.csv",
        "data/HENSMAN_X_2013_May_7.csv",
        "data/HENSMAN_X_2013_May_8.csv",
        "data/HENSMAN_X_2013_May_9.csv",
        "data/HENSMAN_X_2013_May_10.csv",
        "data/HENSMAN_X_2013_May_11.csv",
        "data/HENSMAN_X_2013_May_12.csv",
        "data/HENSMAN_X_2013_May_13.csv",
        "data/HENSMAN_X_2013_May_14.csv",
        "data/HENSMAN_X_2013_May_15.csv",
        "data/HENSMAN_X_2013_May_16.csv",
        "data/HENSMAN_X_2013_May_17.csv",
        "data/HENSMAN_X_2013_May_18.csv",
        "data/HENSMAN_X_2013_May_19.csv",
        "data/HENSMAN_X_2013_May_20.csv",
        "data/HENSMAN_X_2013_May_21.csv",
        "data/HENSMAN_X_2013_May_22.csv",
        "data/HENSMAN_X_2013_May_23.csv",
        "data/HENSMAN_X_2013_May_24.csv",
        "data/HENSMAN_X_2013_May_25.csv",
        "data/HENSMAN_X_2013_May_26.csv",
        "data/HENSMAN_X_2013_May_27.csv",
        "data/HENSMAN_X_2013_May_28.csv",
        "data/HENSMAN_X_2013_May_29.csv",
        "data/HENSMAN_X_2013_May_30.csv",
        "data/HENSMAN_X_2013_May_31.csv",
        "data/HENSMAN_X_2013_June_1.csv",
        "data/HENSMAN_X_2013_June_2.csv",
        "data/HENSMAN_X_2013_June_3.csv",
        "data/HENSMAN_X_2013_June_4.csv",
        "data/HENSMAN_X_2013_June_5.csv",
        "data/HENSMAN_X_2013_June_6.csv",
        "data/HENSMAN_X_2013_June_7.csv",
        "data/HENSMAN_X_2013_June_8.csv",
        "data/HENSMAN_X_2013_June_9.csv",
        "data/HENSMAN_X_2013_June_10.csv",
        "data/HENSMAN_X_2013_June_11.csv",
        "data/HENSMAN_X_2013_June_12.csv",
        "data/HENSMAN_X_2013_June_13.csv",
        "data/HENSMAN_X_2013_June_14.csv",
        "data/HENSMAN_X_2013_June_15.csv",
        "data/HENSMAN_X_2013_June_16.csv",
        "data/HENSMAN_X_2013_June_17.csv",
        "data/HENSMAN_X_2013_June_18.csv",
        "data/HENSMAN_X_2013_June_19.csv",
        "data/HENSMAN_X_2013_June_20.csv",
        "data/HENSMAN_X_2013_June_21.csv",
        "data/HENSMAN_X_2013_June_22.csv",
        "data/HENSMAN_X_2013_June_23.csv",
        "data/HENSMAN_X_2013_June_24.csv",
        "data/HENSMAN_X_2013_June_25.csv",
        "data/HENSMAN_X_2013_June_26.csv",
        "data/HENSMAN_X_2013_June_27.csv",
        "data/HENSMAN_X_2013_June_28.csv",
        "data/HENSMAN_X_2013_June_29.csv",
        "data/HENSMAN_X_2013_June_30.csv",
        "data/HENSMAN_X_2013_July_1.csv",
        "data/HENSMAN_X_2013_July_2.csv",
        "data/HENSMAN_X_2013_July_3.csv",
        "data/HENSMAN_X_2013_July_4.csv",
        "data/HENSMAN_X_2013_July_5.csv",
        "data/HENSMAN_X_2013_July_6.csv",
        "data/HENSMAN_X_2013_July_7.csv",
        "data/HENSMAN_X_2013_July_8.csv",
        "data/HENSMAN_X_2013_July_9.csv",
        "data/HENSMAN_X_2013_July_10.csv",
        "data/HENSMAN_X_2013_July_11.csv",
        "data/HENSMAN_X_2013_July_12.csv",
        "data/HENSMAN_X_2013_July_13.csv",
        "data/HENSMAN_X_2013_July_14.csv",
        "data/HENSMAN_X_2013_July_15.csv",
        "data/HENSMAN_X_2013_July_16.csv",
        "data/HENSMAN_X_2013_July_17.csv",
        "data/HENSMAN_X_2013_July_18.csv",
        "data/HENSMAN_X_2013_July_19.csv",
        "data/HENSMAN_X_2013_July_20.csv",
        "data/HENSMAN_X_2013_July_21.csv",
        "data/HENSMAN_X_2013_July_22.csv",
        "data/HENSMAN_X_2013_July_23.csv",
        "data/HENSMAN_X_2013_July_24.csv",
        "data/HENSMAN_X_2013_July_25.csv",
        "data/HENSMAN_X_2013_July_26.csv",
        "data/HENSMAN_X_2013_July_27.csv",
        "data/HENSMAN_X_2013_July_28.csv",
        "data/HENSMAN_X_2013_July_29.csv",
        "data/HENSMAN_X_2013_July_30.csv",
        "data/HENSMAN_X_2013_July_31.csv",
        "data/HENSMAN_X_2013_August_1.csv",
        "data/HENSMAN_X_2013_August_2.csv",
        "data/HENSMAN_X_2013_August_3.csv",
        "data/HENSMAN_X_2013_August_4.csv",
        "data/HENSMAN_X_2013_August_5.csv",
        "data/HENSMAN_X_2013_August_6.csv",
        "data/HENSMAN_X_2013_August_7.csv",
        "data/HENSMAN_X_2013_August_8.csv",
        "data/HENSMAN_X_2013_August_9.csv",
        "data/HENSMAN_X_2013_August_10.csv",
        "data/HENSMAN_X_2013_August_11.csv",
        "data/HENSMAN_X_2013_August_12.csv",
        "data/HENSMAN_X_2013_August_13.csv",
        "data/HENSMAN_X_2013_August_14.csv",
        "data/HENSMAN_X_2013_August_15.csv",
        "data/HENSMAN_X_2013_August_16.csv",
        "data/HENSMAN_X_2013_August_17.csv",
        "data/HENSMAN_X_2013_August_18.csv",
        "data/HENSMAN_X_2013_August_19.csv",
        "data/HENSMAN_X_2013_August_20.csv",
        "data/HENSMAN_X_2013_August_21.csv",
        "data/HENSMAN_X_2013_August_22.csv",
        "data/HENSMAN_X_2013_August_23.csv",
        "data/HENSMAN_X_2013_August_24.csv",
        "data/HENSMAN_X_2013_August_25.csv",
        "data/HENSMAN_X_2013_August_26.csv",
        "data/HENSMAN_X_2013_August_27.csv",
        "data/HENSMAN_X_2013_August_28.csv",
        "data/HENSMAN_X_2013_August_29.csv",
        "data/HENSMAN_X_2013_August_30.csv",
        "data/HENSMAN_X_2013_August_31.csv",
        "data/HENSMAN_X_2013_September_1.csv",
        "data/HENSMAN_X_2013_September_2.csv",
        "data/HENSMAN_X_2013_September_3.csv",
        "data/HENSMAN_X_2013_September_4.csv",
        "data/HENSMAN_X_2013_September_5.csv",
        "data/HENSMAN_X_2013_September_6.csv",
        "data/HENSMAN_X_2013_September_7.csv",
        "data/HENSMAN_X_2013_September_8.csv",
        "data/HENSMAN_X_2013_September_9.csv",
        "data/HENSMAN_X_2013_September_10.csv",
        "data/HENSMAN_X_2013_September_11.csv",
        "data/HENSMAN_X_2013_September_12.csv",
        "data/HENSMAN_X_2013_September_13.csv",
        "data/HENSMAN_X_2013_September_14.csv",
        "data/HENSMAN_X_2013_September_15.csv",
        "data/HENSMAN_X_2013_September_16.csv",
        "data/HENSMAN_X_2013_September_17.csv",
        "data/HENSMAN_X_2013_September_18.csv",
        "data/HENSMAN_X_2013_September_19.csv",
        "data/HENSMAN_X_2013_September_20.csv",
        "data/HENSMAN_X_2013_September_21.csv",
        "data/HENSMAN_X_2013_September_22.csv",
        "data/HENSMAN_X_2013_September_23.csv",
        "data/HENSMAN_X_2013_September_24.csv",
        "data/HENSMAN_X_2013_September_25.csv",
        "data/HENSMAN_X_2013_September_26.csv",
        "data/HENSMAN_X_2013_September_27.csv",
        "data/HENSMAN_X_2013_September_28.csv",
        "data/HENSMAN_X_2013_September_29.csv",
        "data/HENSMAN_X_2013_September_30.csv",
        "data/HENSMAN_X_2013_October_1.csv",
        "data/HENSMAN_X_2013_October_2.csv",
        "data/HENSMAN_X_2013_October_3.csv",
        "data/HENSMAN_X_2013_October_4.csv",
        "data/HENSMAN_X_2013_October_5.csv",
        "data/HENSMAN_X_2013_October_6.csv",
        "data/HENSMAN_X_2013_October_7.csv",
        "data/HENSMAN_X_2013_October_8.csv",
        "data/HENSMAN_X_2013_October_9.csv",
        "data/HENSMAN_X_2013_October_10.csv",
        "data/HENSMAN_X_2013_October_11.csv",
        "data/HENSMAN_X_2013_October_12.csv",
        "data/HENSMAN_X_2013_October_13.csv",
        "data/HENSMAN_X_2013_October_14.csv",
        "data/HENSMAN_X_2013_October_15.csv",
        "data/HENSMAN_X_2013_October_16.csv",
        "data/HENSMAN_X_2013_October_17.csv",
        "data/HENSMAN_X_2013_October_18.csv",
        "data/HENSMAN_X_2013_October_19.csv",
        "data/HENSMAN_X_2013_October_20.csv",
        "data/HENSMAN_X_2013_October_21.csv",
        "data/HENSMAN_X_2013_October_22.csv",
        "data/HENSMAN_X_2013_October_23.csv",
        "data/HENSMAN_X_2013_October_24.csv",
        "data/HENSMAN_X_2013_October_25.csv",
        "data/HENSMAN_X_2013_October_26.csv",
        "data/HENSMAN_X_2013_October_27.csv",
        "data/HENSMAN_X_2013_October_28.csv",
        "data/HENSMAN_X_2013_October_29.csv",
        "data/HENSMAN_X_2013_October_30.csv",
        #"HENSMAN_X_2013_October_31",
    ],
}
       