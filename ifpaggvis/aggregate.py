import pandas as pd
import tqdm
import numpy as np
import re
from IFPAggVis.ifpaggvis.visualise import colour_based_on_interaction

def summarise_two_interaction_dfs(df1, df2, ligname1, ligname2):
    """ Summarises all interactions of two dfs with interaction fingerprints.
    
    If an interaction is not present in one of both, 0 will be added to 
    indicate absent interaction.
    Note: The column difference to previous is dropped as it is not accurate
    after merging.

    Parameters
    ----------
    df1 : pd.DataFrame()
        IFPs of ligand 1 with occurrence, diff_to_previous as achieved after
        aggregation by structure or time.
    df2 : pd.DataFrame()
        IFPs of ligand 2 with occurence, diff_to_previous as achieved after
        aggregation by structure or time.

    ligname1 : str
        name of e.g. ligand 1 to add to respective rows for df1

    ligname2 : str
        name of e.g. ligand 2 to add to respective rows for df2

    Returns
    -------
    data_order : pd.DataFrame()
        data frame which merged df1 and df2 based on all interactions, 
        which are sorted by residue number with ligname associated to 
        respective rows and information about number of occurence of individual
        interaction fingerprint.
    """
    # Generate vector to note which IFP originates from which ligand
    lig_occ = len(df1) * [ligname1] + len(df2) * [ligname2]

    # Merge two df, add 0 if interaction not present in one of them
    col1 = df1.columns.values.tolist()
    col2 = df2.columns.values.tolist()
    print("Number of columns df1: ",len(col1))
    print("Number of columns df2: ",len(col2))
    result = pd.concat([df1, df2], ignore_index=True, sort=False)
    result.fillna(0,inplace=True)
    print("Number of columns after merge: ", 
          len(result.columns.values.tolist()))

    # Sort columns based on residue number after merging and order df for 
    # further processing
    data = result.loc[:, ~result.columns.isin(['diff_to_prev','occurence'])]
    sorted_cols = sorted(data.columns.values.tolist(),
                         key=lambda s: int(re.search(r'\d+', s).group()))
    data_order = data[sorted_cols].copy()
    data_order.loc[:, 'occurence'] = result['occurence']
    data_order.loc[:, 'Lig'] = lig_occ

    return data_order

def summarise_interactions(interactions):
    """ Summarises all interactions of given to one list.

    Parameters
    ----------
    interactions : list()
        List of column headers from processed interaction data frames.

    Returns
    -------
    interactions_order : list()
        List with merged (i.e. unique) and sorted interactions. Sorted by
        residue number.
    """

    # Flatten interaction list, use set to only get unique interactions
    # Change data type to list for sorting
    interactions_all = list(set([int for el in interactions for int in el]))
    
    # Remove columns that are not interactions
    interactions_all.remove("diff_to_prev")
    interactions_all.remove("occurence")

    # Sort columns based on residue number
    interactions_order = sorted(interactions_all, key=lambda s: int(re.search(r'\d+', s).group()))

    return interactions_order


def calculate_differences_rows(df):
    """ Calculates number of differences/index of current to previous IFP.
    
    Parameters
    ----------
    df : pd.DataFrame()
        IFPs of ligand 1 with occurrence, diff_to_previous as achieved after
        aggregation by structure or time.

    Returns
    -------
    new_list : list
        list of length len(df) with indices of columns (:class:`np.array`) 
        changing between IFPs (rows of df) 
    list_values : list
        list of length len(df) with number of changes (:class:`np.array`) 
        between IFPs (rows of df)
    """
    # Iterate over IFPs in df
    int_list = range(1, len(df), 1)
    new_list = [np.array([])]
    list_values = [np.array([])]

    for i in tqdm.tqdm(int_list):
        # Calculate difference to previous row
        row = df.iloc[i].values
        row1 = df.iloc[i-1].values
        diffval = np.subtract(row, row1)
        notzero = np.nonzero(diffval)
        # Get number of differences 
        values = len(notzero[0]) 
        list_values.append(values)
        # Get indices of nodes with differences
        new_list.append(notzero[0].tolist()) 
        i+=1    
    return new_list, list_values
    

def calculate_differences_rows_agg_frame(df, agg_frame):
    """ Calculates number of differences of IFPs in df to aggregated IFP.
    
    Parameters
    ----------
    df : pd.DataFrame()
        IFPs of ligand

    agg_frame : list
        IFP of aggregated frame

    Returns
    -------
    new_list : list
        list of length len(df) with indices of columns (:class:`np.array`) 
        with difference between IFPs and aggregated frame
    """
    # Iterate over IFPs in df
    int_list = range(1, len(df), 1)
    new_list = [np.array([])]

    for i in tqdm.tqdm(int_list):
        # Calculate difference to aggregated IFP
        row = df.iloc[i].values
        diffval = np.subtract(agg_frame, row)
        notzero = np.nonzero(diffval)
        # Get indices of nodes with differences
        new_list.append(notzero[0])
        i+=1

    return new_list

def summarise_df(df, col_with_diffs, new_col_name="occurence"):
    """ Summarises IFPs in df to temporal aggregated IFP.
    
    All IFPs which are identical and occur after each other, are summarised to
    one IFP and number of IFPs summarised (occurrence) is saved.
    
    Parameters
    ----------
    df : pd.DataFrame()
        IFPs of ligand which should be summarised

    col_with_diffs : str
        Name of column with number of differences to previous row
        
    new_col_name : str
        new column name with number of summarised IFPs (number of occurence),
        default is "occurence"
        
    Returns
    -------
    df_result : pd.DataFrame()
        summarised IFPs by time (temporal aggregation) with number of IFPs
        aggregated to one IFP (occurence)
    """
    # Summarise indices where difference to previous IFP is greater than 0
    diff_idxs = np.where(df[col_with_diffs] != 0)[0]
    # Add first unique IFP to list, first one is always unique
    diff_indices = np.insert(diff_idxs, 0, 0, axis=0)
    print("Detected different frames aggregated by time: ", diff_indices.shape)
    # Iterate over all indices which have differences to previous frame
    indices_to_keep = []
    counted_occcurences = []
    i = 0
    while i < (len(diff_indices)) :

        # Check if not the second last index
        if i != (len(diff_indices)-1):
            # Check indices of IFPs which are kept
            indices_to_keep.append(diff_indices[i])
            # Calculate occurrence of individual IFPs 
            occurence = diff_indices[i+1] - diff_indices[i]
            counted_occcurences.append(occurence)

        # To determine occurence of last ifp, len(df) needs to be substracted
        # from last found element
        else:
            # Check indices of IFPs which are kept
            indices_to_keep.append(diff_indices[i])
            # Calculate occurrence of individual IFPs 
            occurence = len(df) - diff_indices[i]
            counted_occcurences.append(occurence)
        i+=1
    # Summarise IFPs which are identical and occur after each other to new df
    # and save number of occurrence to new column
    df_result = df.loc[indices_to_keep]
    df_result[new_col_name] = counted_occcurences
    return df_result

def calculate_lengths_interaction(df):
    """ Calculate how long individual interaction is present over time.
    
    For each interaction, determine how long across df the interaction is 
    present.
    
    Parameters
    ----------
    df : pd.DataFrame()
        IFPs of ligand
        
    Returns
    -------
     df_dic : dict
        with interacting residue and type as key, and df as value with two 
        columns with presence/absence of interaction and how long they occur
        as absent/present for each individual interaction detected.
        
    colours : dict
        with colours (value, matplotlib.colors.ListedColormap) assigned to 
        interaction type (key, str).
    """
    df_dic = {}
    
    # Search for residues which are interacting with ligand
    interactions = df.columns.values.tolist()
    inter = 0
    residues = []
    # Define initial res_before for following comparison 
    res_before = ""
    while inter < len(interactions):
        search_string = interactions[inter].split("_")[0]
        if search_string != res_before:
            residues.append(search_string)
            res_before = search_string
        inter += 1

    # Define present interaction types per residue, and derive colours 
    # for each interaction type
    colour_int = []
    grouped_interactions = []
    for res in residues:
        # Identify all interactions present per residue and append to list
        inter_list = [x for x in interactions if res in x]
        grouped_interactions.append(inter_list)
        colour_int.append([x2.split("_")[1] for x2 in inter_list])
    flat_int_list = [x3 for s_list in colour_int for x3 in s_list]
    colours = colour_based_on_interaction(set(flat_int_list))

    # Iterate over each interacting residue
    for group_int in grouped_interactions:
        
        # Iterate over each interaction type present for residue        
        interaction_types = []
        values_all_circles = []
        values_plotting = []
        for interaction in group_int:
            inter_type = interaction.split("_")[1]
            interaction_types.append(inter_type)
            
            # Get values of individual column, interaction with residue of
            # certain type
            values = df[interaction].values
            val_before = values[0]
            indices = [0]
            all_values = [val_before]
            i = 1
            i2 = 0
            a = np.array([[0]])
            # Iterate over values in columns
            while i < len(values):
                val = values[i]
                # If value changes then append index, and calculate length of 
                # previous interactions and add to a
                if val != val_before:
                    indices.append(i)
                    length = i - indices[i2]
                    b = np.array([[length]])
                    a = np.append(a, b, axis=0)
                    all_values.append(val_before)
                    val_before = val
                    i2 += 1
                # Check if last position is recorded, otherwise add last value
                # for correct normalisation in pie chart
                if (i == (len(values) -1 )) and (indices[-1] != len(values)):
                    end_length = len(values) - indices[-1]
                    indices.append(i)
                    b = np.array([[end_length]])
                    a = np.append(a, b, axis=0)
                    all_values.append(val_before)
                i+=1
            # Append lengths (np.array) of interaction to list
            values_all_circles.append(a)
            # Add values as list to new list
            values_plotting.append(all_values)
        # Iterate over interaction lengths of each interaction type per residue
        counter = 0
        while counter < len(values_plotting):
            vals = values_all_circles[counter]
            # Get interacting residue and merge with interaction type
            res_type_name = group_int[0].split("_")[0] + "_" + interaction_types[counter]
            # Generate dictionary with values of interaction (present/absent) 
            # as key, and size (lengths) of interaction as value
            data = {"value" : values_plotting[counter], 
                    "size" : vals.flatten()}
            # Generate df from dictionary
            df_result = pd.DataFrame(data)
            # Add df to dictionary as value, and interacting residue and type
            # as key
            df_dic[res_type_name] = df_result
            counter += 1
            
    return df_dic, colours

def get_most_common_ifps(ifps, occurrence, lig_index):
    """ Summarises most common occurring IFP between two ligands.
    
    Analyses list with similar/identical IFPs of two ligands to identify 
    which ones occur most commonly in a given list. Returns the number 
    of occurrence, as well as IFP number to identify correct NW plot for 
    each ligand.
    
    Parameters
    ----------
    ifps : list of dictionary
        IFPs of ligand 1 (key) with IFPs of ligand 2 (values) which belong
        to certain similarity class
    occurrence : list of int
        Occurrence of individual IFPs of ligand 1 and ligand 2.

    lig_index : int
        Index in merged df where ligand 2 starts to correct IFP numbers

    Returns
    -------
    index_max_both : int
        Index where maximum of IFP (for ligand 1 and 2) are represented i.e.
        group number for similarity class analysed. Can be used to access the
        information saved in the dictionaries returned.
        
    lig1_occs : dict
        Dictionary of IFP of ligand 1 with pair number as key, and a dictionary
        as value. The value dictionary has the occurrence as key, and the IFP 
        number as value.
        
    lig2_occs : dict
        Dictionary of IFP of ligand 2 with pair number as key, and a dictionary
        as value. The value dictionary has the occurrence as key, and the IFP 
        number as value.
    """
    # Iterate over list with IFPs that are within similarity class
    lig1_occs = {}
    lig2_occs = {}
    max_occ_both = 0
    index_max_both = int()
    i = 0
    while i < len(ifps):
        
        # Get occurrence and IFP number of lig 1
        lig1_occs[i] = {occurrence[ifps[i][0]] : ifps[i][0]}
        
        # Iterate over all IFPs of lig 2 that are within similarity class to lig 1
        occ_max = 0
        occ_dic = {}
        
        for lig2_ifps in ifps[i][1]:
            # Correct index number to match networks later
            ifp_number_lig2 = lig2_ifps - lig_index
            # Get occurrence of current IFP for lig 2
            occ = occurrence[lig2_ifps]
            
            # Check which of the IFP in list are most common and occurrence
            if occ_max < occ:
                occ_max = occ
                occ_dic = {int(occurrence[lig2_ifps]):ifp_number_lig2}
                
        # Get occurrence and IFP number of lig 2
        lig2_occs[i] = occ_dic
        
        # Calculate number of IFPs represented for lig 1 and lig 2
        sum_occ = occurrence[ifps[i][0]] + int(occurrence[lig2_ifps])
        
        # Check index of most commonly occurring IFP lig 1 and lig 2
        if max_occ_both < sum_occ:
            max_occ_both = sum_occ
            index_max_both = i
        
        i+=1
     
    return index_max_both, lig1_occs, lig2_occs

