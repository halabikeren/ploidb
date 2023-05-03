#!/usr/bin/env python
# coding: utf-8

# In[5]:


from IPython.display import display
from ete3 import Tree, TreeStyle, Tree, TextFace, add_face_to_node, NodeStyle
import pandas as pd
import numpy as np
import os
from Bio import SeqIO


# In[6]:


family_name = "arecaceae"
genus_name = "phoenix"
#family_name = "hydrangeaceae"
#genus_name = "hydrangea"

family_dir = f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/with_model_weighting/by_family_on_unresolved_ALLMB_and_unresolved_ccdb/{family_name}/"
genus_dir = f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/with_model_weighting/by_genus_on_unresolved_ALLMB_and_unresolved_ccdb/{genus_name}/"

print(family_dir)
print(genus_dir)

tree_suffix = "chromevol/stochastic_mapping/gain_loss_dupl_baseNum_baseNumR/MLAncestralReconstruction.tree"
ploidy_suffix = "chromevol/stochastic_mapping/gain_loss_dupl_baseNum_baseNumR/processed_mappings.csv"


# In[7]:


def get_frequency_based_ploidy_classification(
    taxon_to_polyploidy_support,
    tree_with_internal_names_path,
    polyploidy_threshold = 0.75,
    diploidy_threshold = 0.25,
):  # 0 - diploid, 1 - polyploid, np.nan - unable to determine

    full_tree = Tree(tree_with_internal_names_path, format=1)
    for node in full_tree.traverse():
        if "-" in node.name:
            node.name = "-".join(node.name.split("-")[:-1])

    taxon_to_polyploidy_support["is_polyploid"] = (
        taxon_to_polyploidy_support["polyploidy_frequency"]
        >= polyploidy_threshold
    )
    taxon_to_polyploidy_support["is_diploid"] = (
        taxon_to_polyploidy_support["polyploidy_frequency"] <= diploidy_threshold
    )

    # now add ploidy inference by parent
    in_between_names = taxon_to_polyploidy_support.loc[
        (~taxon_to_polyploidy_support.is_polyploid)
        & (~taxon_to_polyploidy_support.is_diploid),
        "NODE",
    ].tolist()
    taxon_to_polyploidy_support.loc[
        taxon_to_polyploidy_support.NODE.isin(in_between_names), "is_polyploid"
    ] = np.nan
    taxon_to_polyploidy_support.loc[
        taxon_to_polyploidy_support.NODE.isin(in_between_names), "is_diploid"
    ] = np.nan

    node_to_is_polyploid = taxon_to_polyploidy_support.set_index("NODE")[
        "is_polyploid"
    ].to_dict()

    def complement_ploidy_inference_by_parent(taxon: str) -> bool:
        taxon_node = [
            l
            for l in full_tree.traverse()
            if l.name.lower().startswith(taxon.lower())
        ][0]
        taxon_parent_node = taxon_node.up
        if taxon_parent_node is None:
            return np.nan
        parent_name = taxon_parent_node.name
        parent_ploidy_level = node_to_is_polyploid.get(parent_name, np.nan)
        taxon_ploidy_level = (
            parent_ploidy_level if parent_ploidy_level == 1 else np.nan
        )
        return taxon_ploidy_level

    taxon_to_polyploidy_support.loc[
        taxon_to_polyploidy_support.is_polyploid.isna(), "is_polyploid"
    ] = taxon_to_polyploidy_support.loc[
        taxon_to_polyploidy_support.is_polyploid.isna(), "NODE"
    ].apply(
        complement_ploidy_inference_by_parent
    )

    taxon_to_polyploidy_support[
        "ploidy_inference"
    ] = taxon_to_polyploidy_support.apply(
        lambda record: 1
        if record.is_polyploid == 1
        else (0 if record.is_diploid == 1 else np.nan),
        axis=1,
    )
    taxon_to_polyploidy_support.NODE = taxon_to_polyploidy_support.NODE
    return taxon_to_polyploidy_support.set_index("NODE")["ploidy_inference"].to_dict()

############################################################################

def get_ploidy_level_inference(d):
    tree_path = f"{d}{tree_suffix}"
    ploidy_path = f"{d}{ploidy_suffix}"
    if not os.path.exists(ploidy_path):
      return np.nan
    mappings_data = pd.read_csv(ploidy_path)
    node_to_ploidy_level = get_frequency_based_ploidy_classification(taxon_to_polyploidy_support=mappings_data,
                                                                     tree_with_internal_names_path=tree_path)
    return node_to_ploidy_level

############################################################################

def process_tree_with_ploidy_data(d, node_to_ploidy_level):
    tree_path = f"{d}{tree_suffix}"
    if not os.path.exists(tree_path):
      base_tree_path = f"{d}tree.nwk"
      print(base_tree_path)
      tree = Tree(base_tree_path)
      base_counts_path = f"{d}counts.fasta"
      count = int(str(list(SeqIO.parse(base_counts_path, format="fasta"))[0].seq))
      for node in tree.traverse():
        node.add_feature(pr_name="chromosome_count", pr_value=count)
        node.add_feature(pr_name="ploidy_level", pr_value=0)
    else:  
      tree = Tree(tree_path, format=1)
      for node in tree.traverse():
          node_data = node.name.split("-")
          node_name = "-".join(node_data[:-1])
          node_chromosome_count = int(node_data[-1])
          node_ploidy_level = node_to_ploidy_level.get(node_name, np.nan)
          node.name = node_name
          node.add_feature(pr_name="chromosome_count", pr_value=node_chromosome_count)
          node.add_feature(pr_name="ploidy_level", pr_value=node_ploidy_level)
    
    for node in tree.traverse():
        if node.up and node.up.ploidy_level < node.ploidy_level:
            node.add_feature(pr_name="event", pr_value="polyploidization")
        elif node.up and node.up.ploidy_level > node.ploidy_level:
            node.add_feature(pr_name="event", pr_value="diploidization")
        else:
            node.add_feature(pr_name="event", pr_value=np.nan)
            
    return tree


# In[8]:


print(f"processing family {family_name} data")
family_ploidy_data = get_ploidy_level_inference(d=family_dir)
family_tree = process_tree_with_ploidy_data(d=family_dir, 
                                            node_to_ploidy_level=family_ploidy_data)

print(f"processing genus {genus_name} data")
genus_ploidy_data = get_ploidy_level_inference(d=genus_dir)
genus_tree = process_tree_with_ploidy_data(d=genus_dir, 
                                            node_to_ploidy_level=genus_ploidy_data)


# In[9]:

if pd.notna(genus_ploidy_data):
  contradicted_taxa = [node for node in genus_ploidy_data if node in family_ploidy_data and genus_ploidy_data[node] != family_ploidy_data[node]]
else:
  contradicted_taxa = [node for node in genus_tree.get_leaf_names() if family_ploidy_data[node] == 1]
print(f"# contradicted taxa between genus {genus_name} inference and family {family_name} inference = {len(contradicted_taxa):,}")    


# In[10]:


# set is_ploidy_transition per dataset: genista ,acacia and fabaceae

def get_contadicted_leaves(tree, contradicted_taxa):
    contradicted_leaves = []
    for l in tree.get_leaves():
        for taxon in contradicted_taxa:
            if l.name.split("-")[0] == taxon:
                contradicted_leaves.append(l)
    return contradicted_leaves

def get_tree_style(tree, fg_nodes):
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.min_leaf_separation = 6
    
    def my_layout(node):
      node_color = "black" if node in fg_nodes else "lightgrey"
      F = TextFace(f" {node.name}", tight_text=False, fgcolor=node_color)
      if node.is_leaf():
        add_face_to_node(F, node, column=0, position="aligned")
      else:
        add_face_to_node(F, node, column=0)
                 
    ts.layout_fn = my_layout
    
    for n in tree.traverse():
        node_color = "grey" if pd.isna(n.ploidy_level) else ("red" if int(n.ploidy_level)==1 else "blue")
        nstyle = NodeStyle()
        nstyle["size"] = 0.01
        if n in fg_nodes:
            nstyle["fgcolor"] = node_color
            nstyle["hz_line_color"] = "black"
            nstyle["vt_line_color"] = "black"
        else:
            nstyle["bgcolor"] = "white"
            nstyle["fgcolor"] = node_color
            nstyle["hz_line_color"] = "lightgrey"
            nstyle["vt_line_color"] = "lightgrey"
            # n.name = ""
        if pd.notna(n.event):
            nstyle["hz_line_color"] = "red" if n.event == "polyploidization" else "green"
            nstyle["hz_line_type"] = 0 if n.event == "polyploidization" else 1
            nstyle["vt_line_color"] = "black"
            nstyle["fgcolor"] = "black"
        nstyle["size"] = 5
        n.set_style(nstyle)
    return ts

def get_nodes_to_keep(tree, contradicted_leaves):
    event_nodes = []
    contradicted_nodes = contradicted_leaves.copy()
    for l in contradicted_leaves:
        curr = l
        parent = l.up
        while parent is not None:
            contradicted_nodes.append(parent)
            if pd.notna(curr.event):
                break
            curr = parent
            parent = parent.up
    contradicted_nodes = list(set(contradicted_nodes))
    return contradicted_nodes

def get_mrca(tree, nodes_to_keep, name: str = ""):
    leaves_to_keep = [l.name for l in nodes_to_keep if l.is_leaf()]
    for node in nodes_to_keep:
        if not node.is_leaf():
            node_leaves = node.get_leaves()
            leaves_to_add = [node_leaves[0]]
            leaves_to_add = node_leaves
            leaves_to_keep += [l.name for l in leaves_to_add]
    leaves_to_keep = list(set(leaves_to_keep))
    tree.prune(leaves_to_keep, preserve_branch_length=True)
    print(f"# leaves in MRCA tree of the contradiceted leaves = {len(tree.get_leaf_names()):,}")
    return tree

def plot_tree_with_events(tree, contradicted_taxa, name):
    contradicted_leaves = get_contadicted_leaves(tree=tree, contradicted_taxa=contradicted_taxa)
    nodes_to_keep = get_nodes_to_keep(tree=tree, contradicted_leaves=contradicted_leaves)
    contradicted_tree = get_mrca(tree=tree, nodes_to_keep=nodes_to_keep, name=name)
    ts = tree_style=get_tree_style(contradicted_tree, nodes_to_keep)
    for node in contradicted_tree.traverse():
        node.name = f"{'' if not node.is_leaf() else node.name+' '} ({node.chromosome_count})"
    contradicted_tree.write(outfile=f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/pruned_tree_{name}.nwk", format=1)
    contradicted_tree.render(f"/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/{name}_tree.png", tree_style=ts)


# In[11]:


print(f"plotting genus {genus_name} tree")
plot_tree_with_events(tree=genus_tree.copy(),
                      contradicted_taxa=contradicted_taxa,
                      name=genus_name)

print(f"plotting family {family_name} tree")
plot_tree_with_events(tree=family_tree.copy(),
                      contradicted_taxa=contradicted_taxa,
                      name=family_name)

