import pandas as pd

def get_res_names_in_col_index(df):
    """ Modifies Multiindex of df
    
    The data frame derived from run_ifp from ProLif, if return_atoms == False 
    is modified to have a single index for further processing. Returns a 
    dataframe with a column for each interaction pair

    Parameters
    ----------
    df : pd.DataFrame()
      Dataframe after Prolif run IFP return_atoms = False

    Returns:
    ----------
    df_new : pd.DataFrame()
          modified dataframe with residue names and type of interaction as 
          columns
    """
    # Iterates over columns in df and appends multiindex 
    # (residue name protein + interaction type) to one string
    new_columns = []
    for el in df.columns:
        new = str(el[0]) + "_" + str(el[1])
        new_columns.append(new)

    # remove multiindex
    df_res = df.droplevel("protein", axis=1)
    df_res.columns=new_columns
    return df_res

def get_interaction_names(interacting_residues, number_nodes, ligand):
    labeldict = {}
    for node in number_nodes:
        labeldict[interacting_residues[node]] = node
    # Change key/value in labeldict
    labeldict_int = {}
    int_type = {}
    for number, inter in labeldict.items():
        # split into residue and interaction type
        labeldict_int[inter] = number.split("_")[0]
        int_type[inter] = number.split("_")[-1]

    labeldict_int[ligand] = ligand # add ligand to one position
    return labeldict_int, int_type

def define_existing_edge_in_IFP(df, ligand, interacting_nodes, max_val,
                                min_val):

    # Check which edge exist in which IFP
    edge_list = []
    i = 0
    # define dic for node colouring
    node_list_all = {}
    while i < len(df):
        edge_time = []
        all_colours = {}
        # node for lig, add so that this node is also coloured
        all_colours[ligand] = [ligand] 
        all_nodes = []

        for node in interacting_nodes:
            if df.iloc[i][node] == True:
                edge_time.append((ligand, node))

                all_nodes.append(node)
        # add edge times to list
        edge_list.append(edge_time)
        # add colouring to dictionary
        all_colours[ligand] = all_nodes
        node_list_all[i] = all_colours # for each time step
        i+=1

    return node_list_all, edge_list

def evaluate_sim_values(values, lig1, lig2, df):

    between_ligs = "LG"+str(lig1)+"_LG"+str(lig2)

    lig1_lig2_k, lig1_lig2_v = flatten_indices(values, between_ligs)
    
    df_mixed_lig1 = df.loc[df.index[lig1_lig2_k]]
    df_mixed_lig2 = df.loc[df.index[lig1_lig2_v]]
    
    df_mixed = pd.concat([df_mixed_lig1, df_mixed_lig2], ignore_index=True, sort=False)

    return df_mixed

def flatten_indices(values,selection):
    
    lig_key = [x[0] for x in values[selection]]
    lig_value = [x[1].tolist() for x in values[selection]]
    
    lig_value_f = [x for smalllist in lig_value for x in smalllist]
    
    unique_keys = sorted(set(lig_key))
    unique_val = sorted(set(lig_value_f))
    
    return unique_keys, unique_val
