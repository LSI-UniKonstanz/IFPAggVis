import matplotlib.pyplot as plt
import matplotlib.markers as pltmarkers
import matplotlib.transforms as mtransforms
import numpy as np
import matplotlib.patches as mpatches
import imageio.v3 as iio
import networkx as nx
from tqdm import tqdm
import pickle

def determine_coordinates_for_plotting(connected_ifps, indices_ligands, lig1, 
                                       lig2, height1, height2):
    """ Calculates which data points need to be connected on plot.

    Parameters
    ----------
    connected_ifps : dict
        IFPs with connection within or between ligand
    indices_ligands: list
        List of lists with start and end indices (:class:`int`) of ligands in 
        df
    lig1 : int
        Iterator variable to check which ligand is considered as 1
    lig2 : int
        Iterator variable to check which ligand is considered as 2
    height1 : int
        y_coord (height) of starting point of line
    height2 : int
        y_coord (height) of end point of line

    Returns
    ------
    coordinates : list
        List of lists with coordinates (:class:`int`) which IFPs need to be 
        connected
    """
    coordinates = []
    # Iterate over dictionary with connected data points
    for vals in connected_ifps:

        # Check if key comes from first ligand, otherwise correct value for 
        # plotting
        if ((vals[0] <= indices_ligands[0][-1]) and
            (lig1 == 0)):
            # All three conditions are true, it is within ligand 1
            x_val1 = vals[0]
        else:
            x_val1 = vals[0] - indices_ligands[lig2][0]

        # Iterate over points which are connected to first point
        for point in np.array(vals[1]).tolist():
            # Check if point is within ligand 1, otherwise correct value for
            # plotting
            if (point <= indices_ligands[0][-1] and 
                (lig1 == 0) and 
                (lig2 == 0)):
                # All three conditions are true, it is within ligand 1
                x_val2 = point
            else:
                x_val2 = point - indices_ligands[lig2][0]
                
            # Assign plotting coordinates to values and add to list
            x_values = [x_val1, x_val2]
            y_values = [height1, height2]
            coordinates.append([x_values, y_values])
            
    return coordinates



def plot_similarity_between_ligands(a, identical_ifps, similar_ifps, 
                                    lig_names, fontsize=14):
    """ Plots identical and similar IFPs between two sets

    Parameters
    ----------
    a : np.array()
        with df["Lig"] column which saves information which IFP belongs to 
        which of the two ligands evaluated


    identical_ifps : dict
        Dictionary of IFPs which are identical to each other

    similar_ifps : dict
        Dictionary of IFPs which are similar to each other

    lig_names : list
        List of ligand names (:class:`str`) where distances should be plotted
        
    fontsize : int
        Size of font in figure, default is 14
        
    Returns
    -------
    plt.figure()
    """
    # set font sizes for plot
    plt.rc('font', size=fontsize)

    # Get start and end index for both ligands
    indices_obs = [] 
    for lig in lig_names:
        indices = np.where(a == lig)
        indices_obs.append([indices[0][0], indices[0][-1]])
    
    # Define matplotlib figure, colours and labels
    fig, ax = plt.subplots(figsize=(20,5))
    labels = ["a)","b)","c)","d)","e)","f)"]
    ax.set_yticks([0,1,2,3,4,5])
    ax.set_yticklabels(labels)
    plt.xlabel("IFP")
    cmap = plt.get_cmap('tab20')
    cmap2 = plt.get_cmap('tab20c')
    blue1 = cmap(0)
    blue2 = cmap2(1)
    red = cmap(6)
    cyan = cmap(18)
    
    # Plots lines representing IFPs for ligand 1
    plt.hlines(y=0, xmin=0, xmax=indices_obs[0][-1], linewidth=3, color=blue1)
    plt.hlines(y=1, xmin=0, xmax=indices_obs[0][-1], linewidth=3, color=blue1)
    plt.hlines(y=2, xmin=0, xmax=indices_obs[0][-1], linewidth=3, color=blue1)

    # Plots lines representing IFPs for ligand 2
    max_ind_corrected = indices_obs[1][-1] - indices_obs[1][0]
    plt.hlines(y=3, xmin=0, xmax=max_ind_corrected, linewidth=3, color=blue2)
    plt.hlines(y=4, xmin=0, xmax=max_ind_corrected, linewidth=3, color=blue2)
    plt.hlines(y=5, xmin=0, xmax=max_ind_corrected, linewidth=3, color=blue2)
    
    
    # Nested while loops to iterate over both ligands and generate possible 
    # combinations
    i = 0
    while i < len(lig_names):
        lig1 = lig_names[i]

        i2 = i
        while i2 < len(lig_names):
            lig2 = lig_names[i2]
            
            # Check whether two different ligands are considered
            if i != i2:
                # Determine coordinates of similar/identical IFPs and plot 
                # connection between them
                data_points = determine_coordinates_for_plotting(
                    similar_ifps[lig1+"_"+lig2], indices_obs, i, i2, 2, 3)
                for data in data_points:
                    ax.plot(data[0], data[1], color=red, linestyle="-",
                            linewidth=0.1)               
                data_points_ident = determine_coordinates_for_plotting(
                    identical_ifps[lig1+"_"+lig2], indices_obs, i, i2, 2, 3)
                for data in data_points_ident:
                    ax.plot(data[0], data[1], color=cyan, linestyle="-",
                            linewidth=0.3)

            # Check whether ligand 1 is currently considered
            if ((i == 0) and
                (i2 == 0)):
                # Determine coordinates of similar/identical IFPs and plot 
                # connection between them
                data_lig1_sim = determine_coordinates_for_plotting(
                    similar_ifps[lig1+"_"+lig2], indices_obs,i, i2,1,2)
                for data in data_lig1_sim:
                    ax.plot(data[0], data[1], color=blue1, linestyle="-",
                            linewidth=0.1, alpha=0.3)                
                data_lig1_ident = determine_coordinates_for_plotting(
                    identical_ifps[lig1+"_"+lig2], indices_obs, i, i2, 0, 1)
                for data in data_lig1_ident:
                    ax.plot(data[0], data[1], color=blue1, linestyle="-",
                            linewidth=0.1)

            # Check whether ligand 2 is currently considered
            if ((i == i2) and
                (i != 0)):
                # Determine coordinates of similar/identical IFPs and plot 
                # connection between them
                data_lig2_sim = determine_coordinates_for_plotting(
                    similar_ifps[lig1+"_"+lig2], indices_obs, i, i2, 3, 4)
                for data in data_lig2_sim:
                    ax.plot(data[0], data[1], color=blue2, linestyle="-",
                            linewidth=0.1, alpha=0.3)               
                data_lig2_ident = determine_coordinates_for_plotting(
                    identical_ifps[lig1+"_"+lig2], indices_obs, i, i2, 4, 5)
                for data in data_lig2_ident:
                    ax.plot(data[0], data[1], color=blue2, linestyle="-",
                            linewidth=0.1) 
            i2+=1
        i+=1
        
    plt.tight_layout()        
    
    return fig

def plot_distance_distribution_sim_hist_line(distances, occ_ifps,
                                             comparison_vals, max_val=None,
                                             min_val=0, x_label="Differences", 
                                             comparison_label="Identical IFPs",
                                             cmap=plt.cm.viridis, fontsize=14):
    """ Plots distance or similarity and occurence within one IFP set

    Parameters
    ----------
    distances : list
        List of lists with distances or similarities (:class:`int` or :class:
        `float`) of IFPs with all other IFPs (within one IFP set)

    occ_ifps : list
        List of occurrence (:class:`int`) of each individual IFP

    comparison_vals : list
        List of IFPs (:class:`int` or :class:`float`) that have certain 
        distance or similarity to others

    max_val : int or float
        Maximum value to map colour of matrix to. If None, maximum of 
        distances is used for mapping
        
    min_val : int or float
        Minimum value to map colour of matrix to, default is 0
        
    x_label : str
        Label of axis for plots with distances or similarities, 
        default is "Differences"

    comparison_label : str
        Label of axis for plot with comparison_vals, with IFPs that are 
        compared to each other, default is "Identical IFPs"
        
    cmap : matplotlib.colors.ListedColormap
        For similarity/difference plotting of matrix, default is viridis.

    fontsize : int
        Size of font in figure, default is 14
        
    Returns
    -------
    plt.figure()
    """
    # set font sizes for plot
    plt.rc('font', size=fontsize)
    
    # Check if distance should be mapped to a maximum
    if max_val is None:
       max_val = np.amax(distances)

    # Define figure
    fig, axs = plt.subplots(3, 2, sharex='col', figsize=[10,12], 
                            gridspec_kw={'width_ratios':[3, 1], 
                                         'height_ratios':[1,1,3]})

    # Turn off axis of subplots which are not needed
    axs[0,1].axis('off')
    axs[2,1].axis('off')

    # Line plot of number of occurrence of individual IFP
    index = range(0,len(occ_ifps),1)
    axs[0,0].plot(index, occ_ifps)
    axs[0,0].set_ylabel('Number occurrence')
    # Set maximum y value to 15 % higher than maximum y value
    y_lim = int((np.amax(occ_ifps) / 100) * 15)
    y = y_lim + np.amax(occ_ifps)
    axs[0,0].set_ylim(0,y)

    # Plot difference/similarity between all IFPs (within IFP set) as histogram
    dist_hist = [x for dist in distances for x in dist]
    axs[1,1].hist(dist_hist)
    axs[1,1].set_xlabel(x_label)
    axs[1,1].xaxis.set_tick_params(labelbottom=True)

    # Plot similarity within IFP set as line plot, represent IFPs as 
    # horizontal line
    axs[1,0].axhline(y=1, xmin=0, xmax=len(occ_ifps), linewidth=3, 
                     color="tab:blue")
    axs[1,0].axhline(y=2, xmin=0, xmax=len(occ_ifps), linewidth=3, 
                     color="tab:blue")
    labels = ["a)","b)"]
    axs[1,0].set_yticks([1,2])
    axs[1,0].set_yticklabels(labels)

    # Iterate over identical/similar values, connect identical/similar IFPs 
    # by vertical lines
    for val in comparison_vals:
        for point in np.array(val[1]).flatten().tolist():
            x_values = [val[0], point]
            y_values = [1, 2]
            axs[1,0].plot(x_values, y_values, linestyle="-", linewidth=0.1, 
                          color="tab:blue")
    axs[1,0].set_ylabel(comparison_label)
    axs[1,0].set_ylim(0.9,2.1)


    # Plot difference/similarity between all IFPs (within IFP set) as matrix
    im = axs[2,0].matshow(distances, origin='upper', aspect='equal', 
                          vmin=min_val, vmax=max_val, cmap=cmap)
    axs[2,0].set_xlabel('IFP')
    axs[2,0].set_ylabel('IFP')
    # Plot colour bar of differences/similarity for matrix
    cb = plt.colorbar(im, ax=axs[2,1], label=x_label)
    cb.ax.yaxis.tick_left()
    axs[2,1].tick_params(right=True)
    cb.ax.yaxis.set_tick_params(labelright=False, labelleft=True)
    cb.ax.yaxis.set_label_position("left")
    plt.tight_layout()
    
    return fig

def get_unique_residue_position(prot_residues, save = True, outfile = "", 
                                sep = "_", split_var = True):
    """ Generates a unique position in graph for each residue.
    
    Parameters
    ----------
    prot_residues: list
        list of residues (:class:`str`) which are interacting with ligand and
        were extracted from df columns.
        
    save: Bool
        If True, dictionary with unique positions is saved to outfile to reload
        for later usage. Default is True.
        
    outfile: str
        Name of file to write unique positions to reload later.
        
    sep: str
        string by which residue name should be split to generate unique 
        residues. Default is "_".
        
    split_var: Bool
        If True, residue name will be split by seperator, 
        in case list of all residues is provided, if false set is generated 
        directly from provided list. Default is True.
        
    Return
    ------
    pos_nodes: dic
        Dictionary with name of residues and ligand (key) and calculated 
        positions (values) 
    
    """
    # Check if residue names need to be split for reading
    if split_var:
        prot_res_unique = list(set([x.split(sep)[0] for x in prot_residues]))
    else:
        prot_res_unique = list(set(prot_residues))
    
    # Define graph for initial node placement
    G = nx.Graph()

    ligand_name = "LIG"
    G.add_node(ligand_name) # Add Ligand node
    G.add_nodes_from(prot_res_unique)
    
    # Connect all nodes with lig
    connect_all = []
    for col_node in prot_res_unique:
        connect_all.append((ligand_name,col_node))
    G.add_edges_from(connect_all)
    
    pos_nodes = nx.drawing.nx_agraph.graphviz_layout(G, prog='neato', 
                                                     args="-Goverlap=scalexy -Gsep=5 -Gscale=2.1")
    
    # save positions to file
    if save:
        with open(outfile + '_unique_residue_position.pkl', 'wb') as fp:
            pickle.dump(pos_nodes, fp)
        
    return pos_nodes

 
def visualise_network_markers(len_df,occurence_ifps, g, pos_nodes, 
                              all_nodes_in_graph, labeldict_int, dic_int_type, 
                              outfile, gif_bool=False, save_nw = False,
                              axes_nw = True, scale_axis=20, 
                              col_lig_node = "white", duration=0.05, 
                              font_size_nodes = 6, dpi=300, 
                              node_size_basic=460):
    """ Plots individual networks as image or gif for each IFP provided.

    Parameters
    ----------
    len_df : int
        Number of interaction fingerprints

    occurence_ifps : list
        List of occurrence (:class:`int`) of each individual IFP

    g : dynetx.DynGraph
        Dynamical graph of all IFPs

    pos_nodes : list
        List of lists with positions (:class:`float` or :class:`int`) of nodes 
        that represent ligand and interacting residues
        
    all_nodes_in_graph : list
        List of lists with interactions (:class:`int`), i.e. residues
        interacting with ligand for each IFP
        
    dic_int_type : dict
        Dictionary of node number which is assigned to interaction type

    labeldict_int : dict
        Dictionary of node number which is assigned to residue

    outfile : str
        Name of outputfile for individual network images without file ending. 
        Please note: Number of images generated = number of IFPs
        
    gif_bool : Bool
        If True, gif of individual image files will be generated, default is 
        False

    save_nw : Bool
        If True, only network (star graph) image will be saved to file without
        plot of occurence, default is False

    axes_nw : Bool
        If True, network image is surrounded by a black box (axes of plot),
        default is True

    scale_axis : int
        Scaling parameter to calculate borders of network plot depending of the
        node positions present. The minimum value (x and y)*scale_axis will be 
        substracted/added to the minimum or maximum value, default of is 20. If 
        the whitespace around the network is too large, decrease this value.

    col_lig_node : str
        Colour of ligand node (center node), default is white

    duration : float
        Duration each image is displayed in gif, default is 0.05 

    font_size_nodes : int
        Font size of node labels, default is 6
        
    dpi : int
        Resolution of exported images, default is 300

    node_size_basic : int
        node sizes and glyph sizes are calculated relative to this value, 
        default is 460. If label is too large for node, this value 
        has to be increased.     
    """
    # Find min/max for network drawing and scale axis according to percentage
    vertexes = list(pos_nodes.values())
    x_list = [vertex [0] for vertex in vertexes]
    
    y_list = [vertex [1] for vertex in vertexes]
    nx_x_min = np.min(x_list) - ( abs((np.min(x_list) * scale_axis)) + node_size_basic + 200 ) 
    nx_x_max = np.max(x_list) + ( abs((np.min(x_list) * scale_axis)) + node_size_basic + 200 ) 
    nx_y_min = np.min(y_list) - ( abs((np.min(x_list) * scale_axis)) + node_size_basic + 200 )
    nx_y_max = np.max(y_list) + ( abs((np.min(x_list) * scale_axis)) + node_size_basic + 200 )

    
    # Define different sizes of nodes/glyphs
    node_size_0 = node_size_basic + 20
    node_size_1 = node_size_basic + 60
    node_size_2 = node_size_basic + 80
    node_size_3 = node_size_basic + 85
    node_size_4 = node_size_basic + 180
    node_size_5 = node_size_basic + 200
    
    # Define glyphs for interaction representation as NW for plotting
    interactions = {}    
    interactions["Anionic"] = {"node_shape":pltmarkers.MarkerStyle(marker=11),
                               "node_size":node_size_3, 
                               "node_color":"tab:blue", "alpha":0.5}
    interactions["Cationic"] = {"node_shape":pltmarkers.MarkerStyle(marker=11),
                                "node_size":node_size_3, 
                                "node_color":"tab:red", "alpha":0.5}
    interactions["CationPi"] = {"node_shape":pltmarkers.MarkerStyle(marker=8),
                                "node_size":node_size_3, 
                                "node_color":"tab:red", "alpha":0.5}
    interactions["PiCation"] = {"node_shape":pltmarkers.MarkerStyle(marker=8),
                                "node_size":node_size_3, 
                                "node_color":"tab:blue", "alpha":0.5}
    interactions["PiStacking"] = {"node_shape":
                                  pltmarkers.MarkerStyle(marker=10), 
                                  "node_size":node_size_3, 
                                  "node_color":"tab:blue", "alpha":0.5}
    interactions["EdgeToFace"] = {"node_shape":pltmarkers.MarkerStyle(marker=9
                                                                      ), 
                                  "node_size":node_size_3, 
                                  "node_color":"tab:red", "alpha":0.5}
    interactions["FaceToFace"] = {"node_shape":pltmarkers.MarkerStyle(marker=9
                                                                      ), 
                                  "node_size":node_size_3, 
                                  "node_color":"tab:blue", "alpha":0.5}
    interactions["Hydrophobic"] = {"node_shape":
                                   pltmarkers.MarkerStyle(marker="o", 
                                                          fillstyle="none"),
                                   "node_size":node_size_2, 
                                   "node_color":"tab:blue", "alpha":0.5}
    interactions["HBAcceptor"] = {"node_shape":
                                  pltmarkers.MarkerStyle(marker="s", 
                                                         fillstyle="none"), 
                                  "node_size":node_size_4, 
                                  "node_color":"tab:blue", "alpha":0.5} 
    interactions["HBDonor"] = {"node_shape":
                               pltmarkers.MarkerStyle(marker="s", 
                                                      fillstyle="none"),
                               "node_size":node_size_4, 
                               "node_color":"tab:red", "alpha":0.5}
    interactions["MetalAcceptor"] = {"node_shape":
                                     pltmarkers.MarkerStyle(marker=2), 
                                     "node_size":node_size_5, 
                                     "node_color":"tab:red", "alpha":0.5}
    interactions["MetalDonor"] = {"node_shape":
                                  pltmarkers.MarkerStyle(marker=2), 
                                  "node_size":node_size_5, 
                                  "node_color":"tab:blue", "alpha":0.5}
    interactions["XBAcceptor"] = {"node_shape":
                                  pltmarkers.MarkerStyle(marker=3), 
                                  "node_size":node_size_5, 
                                  "node_color":"tab:red", "alpha":0.5}
    interactions["XBDonor"] = {"node_shape":pltmarkers.MarkerStyle(marker=3),
                               "node_size":node_size_5, 
                               "node_color":"tab:blue", "alpha":0.5}
    interactions["VdWContact"] = {"node_shape":
                                  pltmarkers.MarkerStyle(marker="o", 
                                                         fillstyle="none"), 
                                  "node_size":node_size_1,
                                  "node_color":"tab:red", "alpha":0.5}

    # Define parameters for ligand visualisation as center node
    options = {"edgecolors": "tab:gray", "node_size": node_size_1}

    # Iterate over all individual IFPs provided for NW visualisation
    frames = np.arange(0, len_df, 1)
    all_images = []
    network_numbers = occurence_ifps.index.tolist()
    number_ifps = occurence_ifps.values
    for frame in tqdm(frames):
        fig, axs = plt.subplots(2, 1, figsize = [5,6], gridspec_kw=
                                {'width_ratios': [1], 'height_ratios': [1,5]})

        # Line plot of number of occurrence of individual IFP
        axs[0].plot(frames, number_ifps,zorder=0)
        axs[0].set_ylabel('Occurence',fontsize='small')
        y_lim = int((np.amax(number_ifps) / 100) * 10)
        y = y_lim + np.amax(number_ifps)
        axs[0].set_ylim(0,y)
        axs[0].set_xlim(0,len(frames))
        axs[0].vlines(ymin=0, ymax=y, x=frame, linewidth=1,linestyle=':',
                      color='r',zorder=5)
        title = "Network number: " + str(network_numbers[frame]) + " occurs " + str(number_ifps[frame]) + " times"
        
        axs[0].set_title(title,fontsize='small', loc='left')

        # Define current IFP network for plotting
        s = g.time_slice(t_from=frame, t_to=frame)
        
        # Select ligand and nodes which are currently interacting with ligand
        lig_node = ["LIG"] #all_nodes_in_graph[frame]["LIG"]
        node_list_all = all_nodes_in_graph[frame]["LIG"]
        
        # draw initial nodes of current IFP network        
        nx.draw_networkx_nodes(s, pos=pos_nodes, ax=axs[1], nodelist=lig_node,
                               node_color=col_lig_node, **options)
        nodes_white = nx.draw_networkx_nodes(s, pos=pos_nodes, ax=axs[1], 
                                             nodelist=node_list_all, 
                                             node_color=col_lig_node,
                                             node_size=node_size_basic)
        nodes_white.set_edgecolor('white') 
        
        # Add glyphs for respective interaction residues depending on 
        # interaction type
        if len(node_list_all) > 0:
            for node in node_list_all:
                plot_conditions = interactions[dic_int_type[node]]
                nx.draw_networkx_nodes(s, pos=pos_nodes, ax=axs[1], 
                                       nodelist=[node], **plot_conditions)
        # Add white node to hide borders of individual glyphs and add edges
        nodes_white = nx.draw_networkx_nodes(s, pos=pos_nodes, ax=axs[1], 
                                             nodelist=node_list_all,
                                             node_color=col_lig_node, 
                                             node_size=node_size_0)
        nodes_white.set_edgecolor('white')
        nx.draw_networkx_edges(s, pos=pos_nodes, ax=axs[1], width=1.0, 
                               alpha=0.5)
        
        # Generate and plot correct labels for nodes / protein interaction
        all_labels = [lig_node, node_list_all]
        labels_flat = [item for sublist in all_labels for item in sublist]
        dict_labels = {}
        [dict_labels.update({x:labeldict_int[x]}) for x in labels_flat]
        nx.draw_networkx_labels(s, pos_nodes, dict_labels, font_size_nodes, 
                                font_color="black")
        # Limit axis of network plot
        axs[1].set_xlim(nx_x_min, nx_x_max)
        axs[1].set_ylim(nx_y_min, nx_y_max)
        
        # Check if axes of nw plot should be turned off
        if not axes_nw:
            axs[1].axis('off')
        
        # Check if only nw plot should be saved to file without occurence plot
        if save_nw:
            plt.tight_layout()
            fig.savefig(outfile + "_frame_" + str(frame) + "_nw.svg", 
                        bbox_inches=mtransforms.Bbox([[0.1, 0], [0.95, 0.70]])
                        .transformed((fig.transFigure - fig.dpi_scale_trans)), 
                        dpi=dpi)
        else:
            plt.tight_layout()
            plt.savefig(outfile + "_frame_" + str(frame) + ".svg", dpi=dpi)

        plt.close('all')
        
    # if gif wanted, export to gif
    if gif_bool == "True":
        for frame in tqdm(frames):
            all_images.append(iio.imread(outfile + "_frame_" + str(frame) 
                                         + ".svg"))
        iio.imwrite(outfile+".gif", all_images, duration=duration)
     
def colour_based_on_interaction(interactions, cmap=plt.cm.viridis):
    """ Generates colours out of cmap for list of len(interactions)

    Can be also used to determine colours for pre-defined list of interactions.
    
    Parameters
    ----------
    interactions : list
        List of interactions (:class:`str`)

    cmap : matplotlib.colors.ListedColormap
        For colour generation, default is viridis.

     Returns
    -------
    colour_dic : dict
        Dictionary with interactions mapping to colour.
        
    """    
    # If list provided is empty, take pre-defined interaction list
    # also to fix colour to certain interaction.
    if len(interactions) == 0:
        interactions = ['Anionic', 'CationPi', 'Cationic', 'EdgeToFace', 
                        'FaceToFace', 'HBAcceptor', 'HBDonor', 'Hydrophobic', 
                        'Interaction', 'MetalAcceptor', 'MetalDonor', 
                        'PiCation', 'PiStacking', 'XBAcceptor', 'XBDonor',
                        'vdWContact']
    # Determine number of colours
    steps = np.linspace(0,1, num=len(interactions))
    colours = [cmap(x) for x in steps]
    # Generate dictionary with mapping interaction to certain colour
    i = 0
    colour_dic = {}
    for val in interactions:
        colour_dic[val] = colours[i]
        i += 1
    return colour_dic

def plot_circle_visualisation(dfs, interactions, colours, outfile, fontsize=14,
                              dpi=300):
    """ Plots interactions within IFP set as circle plot

    Parameters
    ----------
    dfs : list
        List of lists with distances or similarities (:class:`int` or :class:
        `float`) of IFPs with all other IFPs (within one IFP set)

    interactions : list
        List of interactions (:class:`str`) within IFP set

    colours : dict
        Dictionary with interactions mapping to colour

    outfile : str
        Name of outputfile for individual circular plots without file ending. 
        Please note: Number of images generated = number of interactions.
        
    fontsize : int
        Size of font in figure, default is 14

    dpi : int
        Resolution of exported images, default is 300
    """
    # set font sizes for plot
    plt.rc('font', size=fontsize)    

    # Determine individual residues which interact
    inter = 0
    residues = []
    res_before = interactions[inter]

    while inter < len(interactions):
        search_string = interactions[inter].split("_")[0]
        # If interaction is new, save to list
        if search_string != res_before:
            residues.append(search_string)
            res_before = search_string
        inter += 1

    # Check which interactions should appear in same graph (per residue)
    grouped_interactions = []
    grouped_indices = []
    for res in residues:
        i = 0
        index_list = []
        inter_list = []
        for x in interactions:
            if res in x:
                inter_list.append(x)
                index_list.append(i)
            i += 1
        grouped_indices.append(index_list)
        grouped_interactions.append(inter_list)

    # Plot interactions summarised for each residue
    number = 0
    while number < len(grouped_interactions):
        types_plotting = grouped_interactions[number]
        fig, ax = plt.subplots()
        size = 0.1 # thickness of cake slice
        number2 = 0
        # Iterate over individual interactions present for residue
        for interaction_res_type in types_plotting:
            # get interaction column from df
            df = dfs[interaction_res_type]
            vals = df["size"].values
            val_interaction = df["value"].values
            interaction_res  = interaction_res_type.split("_")[0]
            interaction_type = interaction_res_type.split("_")[-1]
            # If interaction is present, colour according to dic, if absent
            # colour lightgrey.
            colours_plot = [colours[interaction_type] if i == 1 
                            else "lightgrey" for i in val_interaction]
            # Determine position of interaction circle and plot interaction
            radius = 1 + (size * number2)
            ax.pie(vals.flatten(), radius=radius, startangle = 90, 
                   colors = colours_plot, 
                   wedgeprops = dict(width = size, edgecolor='w'),
                   counterclock=False)

            number2 += 1
        number += 1
        
        # Set properties of image and save to file
        title = interaction_res
        ax.set(aspect="equal")
        ax.set_title(label=title, fontsize=fontsize, pad=20)
        plt.savefig(outfile + "_pie_" + title + ".svg", dpi=dpi)
        plt.close('all')
    
    # Save legend with colour mapping to file
    fig, ax = plt.subplots(figsize=(2,2))
    ax.axis('off')
    ax.set_xlim(0,0.05)
    ax.set_ylim(0,0.05)
    patches = []
    for key, val in colours.items():
        patch = mpatches.Patch(color = val, label = key)
        patches.append(patch)
    plt.legend(handles=patches, loc = "center", fontsize=fontsize,
               frameon=False)
    plt.tight_layout()
    plt.savefig(outfile + "_pie_legend.svg", dpi=dpi)
    plt.close('all')


