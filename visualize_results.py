import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import networkx as nx
import re
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse

def parse_flowsite_results(designed_residues_file, full_sequence_file=None):
    """
    Parse FlowSite output files into a structured format

    Args:
        designed_residues_file: Path to the designed_residue.csv file
        full_sequence_file: Path to the full_sequence.csv file (optional)

    Returns:
        Dictionary containing parsed sequence information
    """
    # Read the designed residues CSV
    design_df = pd.read_csv(designed_residues_file)

    # Get the column headers (residue positions)
    residue_positions = design_df.columns.tolist()

    # Process each design
    designs = []
    for i, row in design_df.iterrows():
        design = {
            'design_number': i + 1,
            'residues': {}
        }

        # Add each residue position and its amino acid
        for pos in residue_positions:
            design['residues'][pos] = row[pos]

        designs.append(design)

    # If full sequence file is provided, add that information
    if full_sequence_file:
        full_seq_df = pd.read_csv(full_sequence_file)
        for i, design in enumerate(designs):
            if i < len(full_seq_df):
                design['full_sequence'] = full_seq_df.iloc[i]['sequence']

    return {
        'designs': designs,
        'residue_positions': residue_positions
    }

def analyze_amino_acid_distribution(flowsite_data):
    """
    Analyze the distribution of amino acids at each position

    Args:
        flowsite_data: Parsed FlowSite data

    Returns:
        Dictionary with amino acid frequencies at each position
    """
    residue_positions = flowsite_data['residue_positions']
    designs = flowsite_data['designs']

    # Initialize the distribution dictionary
    aa_distribution = {}
    for pos in residue_positions:
        aa_distribution[pos] = {}

    # Count amino acid occurrences at each position
    for design in designs:
        for pos, aa in design['residues'].items():
            if aa not in aa_distribution[pos]:
                aa_distribution[pos][aa] = 0
            aa_distribution[pos][aa] += 1

    # Convert counts to frequencies
    num_designs = len(designs)
    for pos in aa_distribution:
        for aa in aa_distribution[pos]:
            aa_distribution[pos][aa] /= num_designs

    return aa_distribution

def visualize_aa_distribution(aa_distribution):
    """
    Create a heatmap visualization of amino acid distribution at each position

    Args:
        aa_distribution: Dictionary with amino acid frequencies

    Returns:
        Matplotlib figure
    """
    # Convert the distribution to a DataFrame for easier plotting
    positions = list(aa_distribution.keys())

    # Get all unique amino acids
    all_aa = set()
    for pos in positions:
        all_aa.update(aa_distribution[pos].keys())
    all_aa = sorted(list(all_aa))

    # Create a matrix for the heatmap
    matrix = np.zeros((len(positions), len(all_aa)))

    # Fill the matrix with frequencies
    for i, pos in enumerate(positions):
        for j, aa in enumerate(all_aa):
            if aa in aa_distribution[pos]:
                matrix[i, j] = aa_distribution[pos][aa]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(positions) * 0.5 + 2))

    # Create heatmap
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')

    # Set axis labels
    ax.set_xticks(np.arange(len(all_aa)))
    ax.set_yticks(np.arange(len(positions)))
    ax.set_xticklabels(all_aa)
    ax.set_yticklabels(positions)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations showing the frequency values
    for i in range(len(positions)):
        for j in range(len(all_aa)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha="center", va="center",
                              color="white" if matrix[i, j] > 0.5 else "black")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

    # Add title
    ax.set_title("Amino Acid Distribution at Designed Positions")

    fig.tight_layout()
    return fig

def create_sequence_logo(aa_distribution):
    """
    Create a sequence logo representation of the amino acid distribution

    Args:
        aa_distribution: Dictionary with amino acid frequencies

    Returns:
        Matplotlib figure
    """
    # Get all positions and sort them
    positions = list(aa_distribution.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=(len(positions) * 0.5 + 2, 4))

    # Define amino acid properties for coloring
    aa_properties = {
        'G': {'color': 'green', 'group': 'small'},
        'A': {'color': 'green', 'group': 'small'},
        'S': {'color': 'green', 'group': 'small'},
        'T': {'color': 'green', 'group': 'small'},
        'C': {'color': 'green', 'group': 'small'},
        'P': {'color': 'yellow', 'group': 'proline'},
        'D': {'color': 'red', 'group': 'negative'},
        'E': {'color': 'red', 'group': 'negative'},
        'N': {'color': 'magenta', 'group': 'polar'},
        'Q': {'color': 'magenta', 'group': 'polar'},
        'H': {'color': 'blue', 'group': 'positive'},
        'K': {'color': 'blue', 'group': 'positive'},
        'R': {'color': 'blue', 'group': 'positive'},
        'V': {'color': 'black', 'group': 'hydrophobic'},
        'I': {'color': 'black', 'group': 'hydrophobic'},
        'L': {'color': 'black', 'group': 'hydrophobic'},
        'M': {'color': 'black', 'group': 'hydrophobic'},
        'F': {'color': 'black', 'group': 'aromatic'},
        'Y': {'color': 'black', 'group': 'aromatic'},
        'W': {'color': 'black', 'group': 'aromatic'}
    }

    # Plot each position
    bar_width = 0.8
    for i, pos in enumerate(positions):
        sorted_aa = sorted(aa_distribution[pos].items(), key=lambda x: x[1], reverse=True)
        bottom = 0

        for aa, freq in sorted_aa:
            color = aa_properties.get(aa, {}).get('color', 'gray')
            ax.bar(i, freq, bar_width, bottom=bottom, color=color, edgecolor='black')

            # Add text label if frequency is significant
            if freq > 0.1:
                ax.text(i, bottom + freq/2, aa, ha='center', va='center',
                        color='white' if color in ['blue', 'black', 'red'] else 'black',
                        fontweight='bold')

            bottom += freq

    # Set labels and title
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels(positions, rotation=90)
    ax.set_ylabel('Frequency')
    ax.set_title('Sequence Logo of Designed Residues')

    # Add legend for amino acid groups
    groups = {}
    for aa, props in aa_properties.items():
        if props['group'] not in groups:
            groups[props['group']] = props['color']

    handles = [plt.Rectangle((0,0), 1, 1, color=color) for group, color in groups.items()]
    labels = list(groups.keys())
    ax.legend(handles, labels, loc='upper right')

    fig.tight_layout()
    return fig

def visualize_design_comparison(flowsite_data):
    """
    Create a comparison visualization of different designs

    Args:
        flowsite_data: Parsed FlowSite data

    Returns:
        Matplotlib figure
    """
    designs = flowsite_data['designs']
    positions = flowsite_data['residue_positions']

    # Create a matrix for the designs
    design_matrix = np.zeros((len(designs), len(positions)), dtype=object)

    # Fill the matrix with amino acids
    for i, design in enumerate(designs):
        for j, pos in enumerate(positions):
            design_matrix[i, j] = design['residues'][pos]

    # Create figure
    fig, ax = plt.subplots(figsize=(len(positions) * 0.8 + 2, len(designs) * 0.5 + 2))

    # Define amino acid properties for coloring
    aa_properties = {
        'G': {'color': '#99ff99', 'group': 'small'},  # Light green
        'A': {'color': '#66ff66', 'group': 'small'},  # Green
        'S': {'color': '#33ff33', 'group': 'small'},  # Bright green
        'T': {'color': '#00cc00', 'group': 'small'},  # Dark green
        'C': {'color': '#ffcc99', 'group': 'small'},  # Light orange
        'P': {'color': '#ffff66', 'group': 'proline'},  # Yellow
        'D': {'color': '#ff6666', 'group': 'negative'},  # Red
        'E': {'color': '#cc0000', 'group': 'negative'},  # Dark red
        'N': {'color': '#ff99ff', 'group': 'polar'},  # Light magenta
        'Q': {'color': '#ff00ff', 'group': 'polar'},  # Magenta
        'H': {'color': '#9999ff', 'group': 'positive'},  # Light blue
        'K': {'color': '#3333ff', 'group': 'positive'},  # Blue
        'R': {'color': '#0000cc', 'group': 'positive'},  # Dark blue
        'V': {'color': '#cccccc', 'group': 'hydrophobic'},  # Light gray
        'I': {'color': '#999999', 'group': 'hydrophobic'},  # Gray
        'L': {'color': '#666666', 'group': 'hydrophobic'},  # Dark gray
        'M': {'color': '#333333', 'group': 'hydrophobic'},  # Very dark gray
        'F': {'color': '#cc99ff', 'group': 'aromatic'},  # Light purple
        'Y': {'color': '#9933ff', 'group': 'aromatic'},  # Purple
        'W': {'color': '#660099', 'group': 'aromatic'}   # Dark purple
    }

    # Create a color matrix
    color_matrix = np.zeros((len(designs), len(positions)), dtype=object)
    for i in range(len(designs)):
        for j in range(len(positions)):
            aa = design_matrix[i, j]
            color_matrix[i, j] = aa_properties.get(aa, {}).get('color', 'white')

    # Plot the cells with colors
    for i in range(len(designs)):
        for j in range(len(positions)):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_matrix[i, j], edgecolor='black'))
            ax.text(j + 0.5, i + 0.5, design_matrix[i, j], ha='center', va='center', fontweight='bold')

    # Set limits and labels
    ax.set_xlim(0, len(positions))
    ax.set_ylim(0, len(designs))
    ax.set_xticks(np.arange(len(positions)) + 0.5)
    ax.set_yticks(np.arange(len(designs)) + 0.5)
    ax.set_xticklabels(positions)
    ax.set_yticklabels([f"Design {i+1}" for i in range(len(designs))])

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Add title
    ax.set_title("Comparison of FlowSite Designs")

    # Add legend for amino acid properties
    groups = {}
    for aa, props in aa_properties.items():
        if props['group'] not in groups:
            groups[props['group']] = props['color']

    handles = [plt.Rectangle((0,0), 1, 1, color=color) for group, color in groups.items()]
    labels = list(groups.keys())
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1))

    fig.tight_layout()
    return fig

def visualize_3d_pocket_from_pdb(pdb_file, residue_positions, ligand_file=None, output_prefix="pocket"):
    """
    Create a 3D visualization of the binding pocket from PDB file

    Args:
        pdb_file: Path to the protein PDB file
        residue_positions: List of residue positions that form the pocket
        output_prefix: Prefix for output files

    Returns:
        Path to the saved visualization
    """
    try:
        from Bio.PDB import PDBParser
        import py3Dmol
    except ImportError:
        return "Error: Biopython and py3Dmol are required for 3D visualization"

    # Parse residue positions from format like A60, A61, etc.
    parsed_positions = []
    for pos in residue_positions:
        if len(pos) >= 2:
            chain = pos[0]
            try:
                resnum = int(pos[1:])
                parsed_positions.append((chain, resnum))
            except ValueError:
                continue

    # Create HTML with py3Dmol
    view = py3Dmol.view(width=800, height=600)
    view.addModel(open(pdb_file, 'r').read(), 'pdb')
    view.setStyle({'cartoon': {'color': 'gray'}})

    # Highlight pocket residues
    for chain, resnum in parsed_positions:
        view.addStyle({'chain': chain, 'resi': resnum},
                     {'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.7}})

    # Add ligand if provided
    if ligand_file and os.path.exists(ligand_file):
        # Add the ligand model
        view.addModel(open(ligand_file, 'r').read(), 'pdb')

        # Style the ligand - assumes ligand atoms are HETATM
        view.setStyle({'hetflag': True},
                      {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.7},
                       'sphere': {'scale': 0.3}})

    view.zoomTo({'chain': parsed_positions[0][0], 'resi': parsed_positions[0][1]})
    view.spin(True)

    # Save to HTML file
    html_file = f"{output_prefix}_3d_vis.html"
    with open(html_file, 'w') as f:
        f.write(view._repr_html_())

    return html_file

def analyze_sequence_conservation(flowsite_data):
    """
    Analyze how conserved each position is across designs

    Args:
        flowsite_data: Parsed FlowSite data

    Returns:
        Dictionary with conservation scores for each position
    """
    aa_distribution = analyze_amino_acid_distribution(flowsite_data)
    conservation = {}

    for pos, aa_freqs in aa_distribution.items():
        # Calculate Shannon entropy
        entropy = 0
        for aa, freq in aa_freqs.items():
            if freq > 0:
                entropy -= freq * np.log2(freq)

        # Normalize to [0, 1] where 1 is fully conserved
        max_entropy = np.log2(len(aa_freqs)) if aa_freqs else 0
        if max_entropy > 0:
            conservation[pos] = 1 - (entropy / max_entropy)
        else:
            conservation[pos] = 1.0

    return conservation

def visualize_conservation(conservation_scores):
    """
    Create a visualization of conservation scores

    Args:
        conservation_scores: Dictionary with conservation scores

    Returns:
        Matplotlib figure
    """
    positions = list(conservation_scores.keys())
    scores = [conservation_scores[pos] for pos in positions]

    fig, ax = plt.subplots(figsize=(len(positions) * 0.5 + 2, 5))

    # Create bar plot
    bars = ax.bar(positions, scores, color='steelblue')

    # Add colored highlighting based on conservation
    for i, bar in enumerate(bars):
        score = scores[i]
        if score > 0.9:
            bar.set_color('darkred')  # Highly conserved
        elif score > 0.7:
            bar.set_color('red')  # Moderately conserved
        elif score > 0.5:
            bar.set_color('orange')  # Somewhat conserved
        else:
            bar.set_color('lightblue')  # Variable

    # Add value annotations
    for i, score in enumerate(scores):
        ax.text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom', rotation=90)

    # Set labels and title
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Conservation Score')
    ax.set_title('Sequence Conservation at Designed Positions')

    # Add a threshold line for high conservation
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.text(len(positions)-1, 0.81, 'High Conservation (0.8)', color='red', ha='right')

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=90)

    fig.tight_layout()
    return fig

# Function to analyze protein-ligand interactions
def analyze_protein_ligand_interactions(protein_pdb, ligand_pdb_or_mol2, designed_residues):
    """
    Analyze interactions between the protein and ligand

    Args:
        protein_pdb: Path to the protein PDB file
        ligand_pdb_or_mol2: Path to the ligand PDB or MOL2 file
        designed_residues: List of designed residue positions

    Returns:
        Dictionary with interaction data
    """
    try:
        from Bio.PDB import PDBParser, NeighborSearch, Selection
        import numpy as np
        import os
    except ImportError:
        return {"error": "Biopython is required for interaction analysis"}

    # Determine file type
    is_mol2 = ligand_pdb_or_mol2.endswith('.mol2')

    # Parse the protein
    parser = PDBParser(QUIET=True)
    protein_structure = parser.get_structure('protein', protein_pdb)

    # Extract ligand from PDB if it's in PDB format
    ligand_atoms = []
    if not is_mol2:
        # This assumes the ligand atoms are marked as HETATM in the PDB
        ligand_structure = parser.get_structure('ligand', ligand_pdb_or_mol2)
        for model in ligand_structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0].startswith('H_'):  # HETATM
                        for atom in residue:
                            ligand_atoms.append(atom)
    else:
        # For MOL2 files, we'll need to use other tools or parse manually
        # Simplified placeholder - in real usage, use a dedicated MOL2 parser
        return {"error": "MOL2 parsing requires additional libraries like RDKit or OpenBabel"}

    # If no ligand atoms were found, return an error
    if not ligand_atoms:
        return {"error": "No ligand atoms found in the provided file"}

    # Extract protein atoms, focusing on designed residues
    designed_res_atoms = []
    designed_residue_objs = []

    # Parse residue IDs from format like A60, A61, etc.
    parsed_designed_residues = []
    for pos in designed_residues:
        if len(pos) >= 2:
            chain_id = pos[0]
            try:
                res_id = int(pos[1:])
                parsed_designed_residues.append((chain_id, res_id))
            except ValueError:
                continue

    # Get the atoms of designed residues
    for model in protein_structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] == ' ':  # Standard amino acid
                    res_id = residue.id[1]
                    if (chain_id, res_id) in parsed_designed_residues:
                        for atom in residue:
                            designed_res_atoms.append(atom)
                        designed_residue_objs.append(residue)

    # Calculate interactions
    interactions = []
    distance_threshold = 4.0  # Angstroms

    # Create NeighborSearch for ligand atoms
    ns = NeighborSearch(ligand_atoms)

    # For each designed residue atom, find nearby ligand atoms
    for residue in designed_residue_objs:
        residue_interactions = []
        for atom in residue:
            nearby_ligand_atoms = ns.search(atom.coord, distance_threshold)
            for ligand_atom in nearby_ligand_atoms:
                distance = np.linalg.norm(atom.coord - ligand_atom.coord)
                interaction_type = classify_interaction(atom, ligand_atom, distance)

                if interaction_type:
                    residue_interactions.append({
                        'protein_residue': f"{residue.get_parent().id}:{residue.id[1]}",
                        'protein_atom': atom.name,
                        'ligand_atom': ligand_atom.name,
                        'distance': float(distance),
                        'interaction_type': interaction_type
                    })

        if residue_interactions:
            interactions.extend(residue_interactions)

    return {
        'interactions': interactions,
        'designed_residues': designed_residues,
        'num_interactions': len(interactions)
    }

def classify_interaction(protein_atom, ligand_atom, distance):
    """
    Classify the type of interaction between protein and ligand atoms

    Args:
        protein_atom: Protein atom object
        ligand_atom: Ligand atom object
        distance: Distance between atoms in Angstroms

    Returns:
        String describing the interaction type
    """
    # This is a simplified classification - in real usage, use more sophisticated methods

    # Get atom names
    p_name = protein_atom.name
    l_name = ligand_atom.name

    # Check for potential hydrogen bonds (simplified)
    if distance <= 3.5:
        # Check if either atom is oxygen or nitrogen (potential H-bond)
        if (p_name.startswith('O') or p_name.startswith('N')) and (l_name.startswith('O') or l_name.startswith('N')):
            return 'hydrogen_bond'

    # Check for potential ionic interactions (simplified)
    if distance <= 4.0:
        # Positively charged protein atoms
        if p_name in ['NH1', 'NH2', 'NZ']:
            # Negatively charged ligand atoms
            if l_name.startswith('O'):
                return 'ionic'
        # Negatively charged protein atoms
        elif p_name in ['OD1', 'OD2', 'OE1', 'OE2']:
            # Positively charged ligand atoms
            if l_name.startswith('N'):
                return 'ionic'

    # Check for hydrophobic interactions (simplified)
    if distance <= 4.0:
        # Carbon atoms potentially involved in hydrophobic interactions
        if p_name.startswith('C') and l_name.startswith('C'):
            return 'hydrophobic'

    # If no specific interaction is identified
    if distance <= 4.0:
        return 'close_contact'

    return None

def visualize_protein_ligand_interactions(interaction_data, output_dir="interaction_analysis"):
    """
    Visualize protein-ligand interactions

    Args:
        interaction_data: Dictionary with interaction data
        output_dir: Directory to save visualization outputs

    Returns:
        Dictionary with paths to the generated visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os

    # Check if there's an error in the interaction data
    if 'error' in interaction_data:
        return {'error': interaction_data['error']}

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outputs = {}

    # Convert interactions to DataFrame for easier plotting
    if not interaction_data['interactions']:
        return {'error': 'No interactions found to visualize'}

    df = pd.DataFrame(interaction_data['interactions'])

    # 1. Heatmap of interactions by residue and type
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Create a pivot table for the heatmap
    pivot_df = pd.crosstab(df['protein_residue'], df['interaction_type'])

    # Plot heatmap
    sns.heatmap(pivot_df, cmap='YlGnBu', annot=True, fmt='d', ax=ax1)
    ax1.set_title('Protein-Ligand Interactions by Residue and Type')
    ax1.set_xlabel('Interaction Type')
    ax1.set_ylabel('Protein Residue')

    output_file = os.path.join(output_dir, "interaction_heatmap.png")
    fig1.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['interaction_heatmap'] = output_file
    plt.close(fig1)

    # 2. Bar chart of interaction counts by residue
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Count interactions by residue
    residue_counts = df['protein_residue'].value_counts().sort_index()

    # Plot bar chart
    sns.barplot(x=residue_counts.index, y=residue_counts.values, ax=ax2)
    ax2.set_title('Number of Interactions by Residue')
    ax2.set_xlabel('Protein Residue')
    ax2.set_ylabel('Number of Interactions')
    plt.xticks(rotation=90)

    output_file = os.path.join(output_dir, "interaction_by_residue.png")
    fig2.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['interaction_by_residue'] = output_file
    plt.close(fig2)

    # 3. Network visualization of interactions
    fig3, ax3 = plt.subplots(figsize=(10, 10))

    # Create network visualization with networkx
    import networkx as nx

    G = nx.Graph()

    # Add protein residue nodes
    for residue in df['protein_residue'].unique():
        G.add_node(residue, node_type='protein')

    # Add ligand atom nodes
    for atom in df['ligand_atom'].unique():
        G.add_node(f"LIG:{atom}", node_type='ligand')

    # Add edges for interactions
    for _, row in df.iterrows():
        G.add_edge(
            row['protein_residue'],
            f"LIG:{row['ligand_atom']}",
            weight=1/row['distance'],
            interaction=row['interaction_type']
        )

    # Set position using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw protein nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n, attr in G.nodes(data=True) if attr.get('node_type')=='protein'],
        node_color='lightblue',
        node_size=500,
        alpha=0.8
    )

    # Draw ligand nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n, attr in G.nodes(data=True) if attr.get('node_type')=='ligand'],
        node_color='lightgreen',
        node_size=300,
        alpha=0.8
    )

    # Color edges by interaction type
    interaction_colors = {
        'hydrogen_bond': 'blue',
        'ionic': 'red',
        'hydrophobic': 'gray',
        'close_contact': 'yellow'
    }

    # Draw edges by interaction type
    for itype, color in interaction_colors.items():
        edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get('interaction')==itype]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=2, alpha=0.7, edge_color=color)

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Add legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Protein Residue'),
        mpatches.Patch(color='lightgreen', label='Ligand Atom')
    ]
    for itype, color in interaction_colors.items():
        legend_elements.append(mpatches.Patch(color=color, label=itype))

    ax3.legend(handles=legend_elements, loc='upper right')
    ax3.set_title('Protein-Ligand Interaction Network')
    ax3.axis('off')

    output_file = os.path.join(output_dir, "interaction_network.png")
    fig3.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['interaction_network'] = output_file
    plt.close(fig3)

    # 4. Distance distribution histogram
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    sns.histplot(df['distance'], bins=20, kde=True, ax=ax4)
    ax4.set_title('Distribution of Interaction Distances')
    ax4.set_xlabel('Distance (Å)')
    ax4.set_ylabel('Count')

    output_file = os.path.join(output_dir, "distance_distribution.png")
    fig4.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['distance_distribution'] = output_file
    plt.close(fig4)

    return outputs

# Main function to process FlowSite output and generate visualizations
def analyze_flowsite_output(designed_residues_file, full_sequence_file=None, protein_pdb=None, ligand_pdb_or_mol2=None, output_dir="flowsite_analysis"):
    """
    Process FlowSite output and generate comprehensive visualizations

    Args:
        designed_residues_file: Path to the designed_residue.csv file
        full_sequence_file: Path to the full_sequence.csv file (optional)
        protein_pdb: Path to the protein PDB file (optional, for 3D visualization)
        ligand_pdb_or_mol2: Path to the ligand PDB or MOL2 file (optional, for interaction analysis)
        output_dir: Directory to save visualization outputs

    Returns:
        Dictionary with paths to the generated visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse FlowSite output
    flowsite_data = parse_flowsite_results(designed_residues_file, full_sequence_file)

    # Generate visualizations
    outputs = {}

    # 1. Amino acid distribution
    aa_distribution = analyze_amino_acid_distribution(flowsite_data)
    fig1 = visualize_aa_distribution(aa_distribution)
    output_file = os.path.join(output_dir, "aa_distribution.png")
    fig1.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['aa_distribution'] = output_file
    plt.close(fig1)

    # 2. Sequence logo
    fig2 = create_sequence_logo(aa_distribution)
    output_file = os.path.join(output_dir, "sequence_logo.png")
    fig2.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['sequence_logo'] = output_file
    plt.close(fig2)

    # 3. Design comparison
    fig3 = visualize_design_comparison(flowsite_data)
    output_file = os.path.join(output_dir, "design_comparison.png")
    fig3.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['design_comparison'] = output_file
    plt.close(fig3)

    # 4. Conservation analysis
    conservation = analyze_sequence_conservation(flowsite_data)
    fig4 = visualize_conservation(conservation)
    output_file = os.path.join(output_dir, "conservation.png")
    fig4.savefig(output_file, dpi=300, bbox_inches='tight')
    outputs['conservation'] = output_file
    plt.close(fig4)

    # 5. 3D visualization (if PDB file provided)
    if protein_pdb and os.path.exists(protein_pdb):
        residue_positions = flowsite_data['residue_positions']
        html_file = visualize_3d_pocket_from_pdb(
            protein_pdb,
            residue_positions,
            ligand_file=ligand_pdb_or_mol2 if ligand_pdb_or_mol2 and os.path.exists(ligand_pdb_or_mol2) else None,
            output_prefix=os.path.join(output_dir, "pocket")
        )
        if isinstance(html_file, str) and not html_file.startswith("Error"):
            outputs['3d_visualization'] = html_file

    # 6. Protein-ligand interaction analysis (if both protein and ligand files provided)
    if protein_pdb and ligand_pdb_or_mol2 and os.path.exists(protein_pdb) and os.path.exists(ligand_pdb_or_mol2):
        # Analyze interactions
        interaction_data = analyze_protein_ligand_interactions(
            protein_pdb,
            ligand_pdb_or_mol2,
            flowsite_data['residue_positions']
        )

        # Visualize interactions
        if 'error' not in interaction_data:
            interaction_vis_outputs = visualize_protein_ligand_interactions(
                interaction_data,
                output_dir=os.path.join(output_dir, "interactions")
            )

            # Add interaction visualization outputs to main outputs
            for key, value in interaction_vis_outputs.items():
                outputs[f'interaction_{key}'] = value

    return outputs

if __name__ == "__main__":
    # Replace these with your actual file paths
    parser = argparse.ArgumentParser(description="Visualization for protein-ligand interaction.")
    parser.add_argument("-d", "--designed_residue", default="data/inference_out/complexid0/designed_residues.csv")
    parser.add_argument("-o", "--outdir", default="data/flowsite_analysis")
    parser.add_argument("-f", "--full_sequence", default="data/inference_out/complexid0/full_sequences.csv")
    parser.add_argument("-p", "--protein_pdb", default="data/2fc2_unit1_protein.pdb")
    parser.add_argument("-l", "--ligand_pdb", default="data/2fc2_HEM_HBI_HAR_NO.pdb")
    args = parser.parse_args()

    designed_residues_file = args.designed_residue
    outdir = args.outdir
    if args.full_sequence:
        full_sequence_file = args.full_sequence
    if args.protein_pdb:
        protein_pdb = args.protein_pdb
    if args.ligand_pdb:
        ligand_pdb_or_mol2 = args.ligand_pdb

    outputs = analyze_flowsite_output(
        designed_residues_file,
        full_sequence_file,
        protein_pdb,
        ligand_pdb_or_mol2,
        output_dir=outdir
    )

    print("Analysis complete. Output files:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")