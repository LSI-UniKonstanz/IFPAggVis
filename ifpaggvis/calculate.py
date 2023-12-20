import numpy as np
import timeit
from sklearn.metrics import pairwise_distances_chunked

def calculate_where_diff_zero(distances, indices):
    """ Calculates which IFPs have no differences
    
    Interaction fingerprints within one ligand are the same,
    returns indices of IFP with distance 0.

    Parameters
    ----------
    distances : numpy.array 
        with calculated number of distances between all IFPs
    indices : pd.df.index.tolist()
        index of pandas df with aggregated IFPs.

    Returns
    -------
    identical_values: list
        List of indices of IFPs (:class:`int`) that have 0 distance to others
    """
    # Check which frames have a distance of 0 to other frames
    identical_values = []
    i = 0
    while i < len(distances):
        # Exclude the same frame with i+1 (here the distance is always 0)
        dist_zeros = np.where(distances[i][i+1:] == 0)[0]
        # Check if a IFP with distance 0 was found
        if len(dist_zeros) > 0:
            # Add i+1 to index, as we slice for i+1 to end val
            indices = dist_zeros + i + 1
            identical_values.append([i, indices])
        i+=1
        
    return identical_values

def iterate_dict_assign_IFPs(dic_distances, indices_obs, i, i2, comparison, 
                             identical_name):
    """ Assign IFPs of categories to between/within ligand category 
    
    returns dictionary with assignment of IFPs which have certain category/
    relation to each other (i.e. identical, similar, dissimilar)

    Parameters
    ----------
    dic_distances : dict
        key (IFP) and which interactions (values) are assigned to it belonging 
        to certain category

    indices_obs : list 
        with list of indices (start/end) of ligands

    i : int
        iterator of first while loop
    
    i2 : int
        iterator of second while loop

    comparison : str
        ligand name 1

    identical_name : str
        ligand name 2

    Returns
    -------
    category_values : dict
        with IFPs belonging to category for within and in between ligands
    """    
    # Iterate over IFPs in dictionary
    obs = []
    obs_within = []
    category_values = {}
    
    for key, value in dic_distances.items():
        # Check if key is in index of first one, since distances were only
        # investigated for everything that follows, not backwards.

        if ((key >= indices_obs[i][0]) and
            (key <= indices_obs[i][1])):

            # check if value exist that is IFP in second one (if different
            # get those that differ between ligands), if i == i2 
            # interaction is within ligand. Sort key/ value based on 
            # category and if interaction within or between ligands
            evaluate = np.asarray((value >= indices_obs[i2][0]) & 
                                  (value <= indices_obs[i2][1]))

            if i == i2:
                val_within = value[evaluate.nonzero()]
                if val_within.size != 0:
                    obs_within.append([key, val_within])

            if i != i2:
                val = value[evaluate.nonzero()]
                if val.size != 0:
                    obs.append([key, val])

    # Add IFPs that are identical between/within ligands to separate
    # dictionarys
    if i == i2:
        category_values[comparison + "_" + identical_name] = obs_within
    if i != i2:              
        category_values[comparison + "_" + identical_name] = obs
    
    return category_values

def calculate_where_diff_and_sim(a, distances, lignames, identical_threshold, 
                                 similarity_threshold, dissimilarity_threshold,
                                 dissimilarity_bool):
    """ Calculate which IFPs are identical/similar/dissimilar 
    
    returns indices of IFPs of different categories.

    Parameters
    ----------
    a : np.array()
        with df["Lig"] column which saves information which IFP belongs to 
        which of the two ligands evaluated

    distances : numpy.array 
        with calculated number of distances between all IFPs

    lignames : list
        ligand names (:class:`str`) where distances should be determined
    
    identical_threshold : float()
        threshold were ligand is considered identical, x >= threshold.

    similarity_threshold : list
        thresholds (:class:`float` or :class:`int`) were ligand is considered 
        similar. Lower threshold >= x < upper threshold,

    dissimilarity_threshold : float()
        thresholds were ligand is considered dissimilar, x < threshold.

    dissimilarity_bool : bool
	    If true, dissimilarity instead of similarity was calculated. Values 
        will be inverted to similarity by calculating 1-dissimilarity value.

    Returns
    -------
    identical_values : dict
        IFPs which are identical to each other

    similar_values : dict
        IFPs which are similar to each other
    
    dissimilar_values : dict
        IFPs which are dissimilar to each other
    """
    dic_similar = {}
    dic_dissimilar = {}
    dic_identical = {}
    # Identify which IFPs are within identical, similar and dissimilar 
    # threshold, save to dic
    # Iterate over distances, increase iterator to consider only one direction 
    # of interaction (undirected)
    i = 0
    while i < len(distances):
        # Correct distance value if dissimilarity was calculated
        # Do not add i+1 here because then you shift the index position by one
        if dissimilarity_bool:
            arr = 1- distances[i][i:]
        else:
            arr = distances[i][i:]
        
        # Check where array has specific values and assign to category, add to
        # dictionary for each IFP
        dist_similar = np.where((arr >= similarity_threshold[0]) & 
                                (arr < similarity_threshold[1]))[0]
        dist_dissimilar = np.where(arr < dissimilarity_threshold[0])[0]
        dist_identical = np.where(arr >= identical_threshold[0])[0] 
        if len(dist_identical) > 0:
            # [1:] to ignore similarity to itself
            indices = dist_identical[1:] + i 
            dic_identical[i] = indices

        if len(dist_similar) > 0:
            indices = dist_similar + i
            dic_similar[i] = indices

        if len(dist_dissimilar) > 0:
            indices = dist_dissimilar + i
            dic_dissimilar[i] = indices
        i+=1
    
    indices_obs = []

    # Identify indices of IFPs of individual ligands
    for lig in lignames:
        indices = np.where(a == lig)
        indices_obs.append([indices[0][0], indices[0][-1]])

    # Identify if similarity/identical values/ dissimilarity is within or in
    # between ligands and save to dictionary
    i = 0
    i2 = 0
    # iterate over individual indices, i.e. over x ligands in nested while loop
    # to compare all combinations with each other
    all_ident_vals = {}
    all_sim_vals = {}
    all_dissim_vals = {}
    while i < len(indices_obs):
        i2 = i
        while i2 < len(indices_obs):
            comparison = lignames[i] #+ "_" + lignames[i2]
            identical_name = lignames[i2] #+ "_" + lignames[i]
            
            # Iterate over individual dictionarys of different categories to 
            # identify matching IFPs
            identical_values = iterate_dict_assign_IFPs(dic_identical, 
                                                        indices_obs, i, i2, 
                                                        comparison, 
                                                        identical_name)
            similar_values = iterate_dict_assign_IFPs(dic_similar, indices_obs,
                                                      i, i2, comparison, 
                                                      identical_name)
            
            dissimilar_values = iterate_dict_assign_IFPs(dic_dissimilar, 
                                                         indices_obs, i, i2, 
                                                         comparison, 
                                                         identical_name)
            
            # add calculated values to dictionary to later return all results
            for key, value in identical_values.items():
                all_ident_vals[key] = value
                
            for key, value in similar_values.items():
                all_sim_vals[key] = value
                
            for key, value in dissimilar_values.items():
                all_dissim_vals[key] = value

            i2+=1
        i+=1
    
    return all_ident_vals, all_sim_vals, all_dissim_vals

def calculate_number_percent_IFP(df_id, df_sim, df_dissim, column, length_org):
    """ Calculates how many and percentage of IFPs are identical, similar or
    dissimilar for two different simulations)

    Parameters
    ----------
    df_id : pd.DataFrame()
        with IFPs identical between two simulations
        
    df_sim : pd.DataFrame()
        with IFPs similar between two simulations

    df_dissim : pd.DataFrame()
        with IFPs dissimilar between two simulations

    column : str()
        column with number of occurrence in df_id, df_sim, df_dissim
        
    length_org : int()
        integer of total number of IFPs compared between simulations
        
    Returns
    -------
    result
        list with number and percentages of identical, similar and dissimilar
        IFPs between two simulations
    """    
    one_percent = (length_org/100)

    number_id = df_id[column].sum()
    percent_id = round(number_id / one_percent, 2)
   
    number_sim = df_sim[column].sum()
    percent_sim = round(number_sim / one_percent, 2)

    number_dissim = df_dissim[column].sum()
    percent_dissim = round(number_dissim / one_percent, 2)
    
    result = [number_id, percent_id, number_sim, percent_sim, number_dissim,
              percent_dissim]
  
    return result

def diff_function(x1, x2):
    """ Calculates number of differences of two IFPs

    Parameters
    ----------
    x1 : numpy.array
        with IFP1
        
    x2 : numpy.array
        with IFP2
        
    Returns
    -------
    dist
        int with number of differences
    """

    diffval = np.subtract(x1, x2)
    notzero = np.nonzero(diffval)
    dist = len(notzero[0])
    
    return dist

def calculate_distances(IFP_values, memory, difference_function=diff_function):
    """ Function to calculate distances between all IFPs in list

    Parameters
    ----------
    IFP_values : numpy.arrays
        list of IFPs as numpy array
        
    memory : int
        integer of available memory on machine to calculate distances
        
    difference_function : str or callable, default=diff_function
        calls pairwise_distances_chunked of scikit-learn. If string, it must
        be an option allowed by scikit-learn.

    Returns
    -------
    distances
        list of integers with number of differences between all IFPs
    """
    X = np.array(IFP_values)
    print("shape array: ", X.shape)
    print(difference_function)
    tic=timeit.default_timer()
    distances = next(pairwise_distances_chunked(X, n_jobs=-1,
                                                metric=difference_function,
                                                working_memory=memory))
    toc=timeit.default_timer()
    print("Time needed for distance calculation: " + str(toc-tic) + " s" )
    
    return distances