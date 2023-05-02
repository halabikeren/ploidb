from ete3 import Tree, TreeStyle, NodeStyle, TextFace, RectFace
import matplotlib.colors as mcolors
import click
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"


@click.command()
@click.option(
    "--tree_path",
    help="path to angiosperms tree with clade and polyploidization index features",
    type=click.Path(exists=True, file_okay=True, readable=True),
    required=False,
    default="/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/ALLMB_angiosperms_tree_with_polyploioidizations.nwk",
)
@click.option(
    "--color_by_cat", help="indicator weather to color by clade /order or not", type=bool, required=False, default=True
)
@click.option(
    "--tree_per_clade",
    help="indicator weather to draw tree per clade separately",
    type=bool,
    required=False,
    default=True,
)
@click.option(
    "--collapse",
    help="indicator weather to collapse clades following the last polyploidization event",
    type=bool,
    required=False,
    default=False,
)
@click.option(
    "--output_dir",
    help="directory to write trees to ",
    type=click.Path(exists=False),
    required=False,
    default="/groups/itay_mayrose/halabikeren/PloiDB/chromevol/results/",
)
def draw_tree(tree_path: str, color_by_cat: bool, tree_per_clade: bool, collapse: bool, output_dir: str):
    tree = Tree(tree_path, format=1)
    tree.get_tree_root().add_feature(pr_name="num_prev_polyploidizations", pr_value=0)
    tree.convert_to_ultrametric()

    poly_index_to_color = {0: "silver", 1: "#a1dab4", 2: "#41b6c4", 3: "#2c7fb8", 4: "#253494", 5: "midnightblue"}

    if collapse:
        for node in tree.traverse("preorder"):
            if int(node.num_prev_polyploidizations) == max(
                [int(l.num_prev_polyploidizations) for l in node.get_leaves()]
            ):
                for child in node.get_children():
                    node.remove_child(child)
    print(f"# polyplodizations = {len([n for n in tree.traverse() if n.name.startswith('polyploidization')]):,}")
    print(f"# leaves = {len(tree.get_leaves()):,}")

    def get_tree_style(tree):
        ts = TreeStyle()
        ts.mode = "c"  # draw tree in circular mode
        ts.scale = 20
        ts.show_leaf_name = False
        ts.show_scale = False
        ts.margin_top = 10
        ts.margin_right = 10
        ts.margin_left = 10
        ts.margin_bottom = 10

        cat_to_color = {}
        cat_name = "order" if tree_per_clade else "order"
        if color_by_cat:
            colors = list(mcolors.TABLEAU_COLORS.keys())
            categories = list(set([l.__dict__[cat_name] for l in tree.get_leaves()]))
            for i in range(len(categories)):
                cat_to_color[categories[i]] = colors[i]
                cat_to_color["nan"] = "white"

        def my_layout(node):
            node_color = poly_index_to_color[int(node.num_prev_polyploidizations)]
            node_clade_color = (
                cat_to_color.get(node.clade, "white")
                if ("clade" in node.__dict__["features"] and type(node.clade) == str)
                else "white"
            )

        ts.layout_fn = my_layout

        for n in tree.traverse():
            node_color = poly_index_to_color[int(n.num_prev_polyploidizations)]
            nstyle = NodeStyle()
            nstyle["size"] = 15 if color_by_cat and n.is_leaf() else 0.01
            nstyle["shape"] = "square"
            nstyle["fgcolor"] = (
                cat_to_color.get(n.__dict__[cat_name], "white") if n.is_leaf() and color_by_cat else node_color
            )
            nstyle["hz_line_color"] = node_color
            nstyle["vt_line_color"] = node_color
            nstyle["hz_line_type"] = 0
            nstyle["hz_line_type"] = 0
            nstyle["vt_line_width"] = 5
            nstyle["hz_line_width"] = 5
            nstyle["bgcolor"] = "white"
            n.set_style(nstyle)

        if color_by_cat:
            for cat in cat_to_color:
                color = cat_to_color[cat]
                ts.legend.add_face(face=RectFace(10, 10, fgcolor="white", bgcolor="white", label=""),
                                   column=20)
                ts.legend.add_face(
                    face=RectFace(
                        50,
                        50,
                        fgcolor="white",
                        bgcolor=color,
                        label={
                            "fontsize": 20,
                            "text": f"{cat}{''.join([' ']*(30-len(cat)))}",
                            "color": "black" if cat != "nan" else "white",
                        },
                    ),
                    column=20,
                )
        for poly_index in poly_index_to_color:
            text = f"{poly_index}th polyploidization"
            ts.legend.add_face(
                face=RectFace(
                    10,
                    1,
                    fgcolor="white",
                    bgcolor=poly_index_to_color[poly_index],
                    label={
                        "fontsize": 20,
                        "text": f"{text}{''.join([' ']*(30-len(text)))}",
                        "color": "black",
                    },
                ),
                column=20,
            )
        return ts

    if tree_per_clade:
        clades = set([l.clade for l in tree.get_leaves()])
        for clade in clades:
            if clade == "nan":
                continue
            clade_tree = tree.copy()
            clade_leaves = [l for l in clade_tree.get_leaves() if l.clade == clade]
            clade_tree.prune(clade_leaves, preserve_branch_length=True)
            print(f"# leaves in {clade} = {len(clade_leaves):,}")
            ts = get_tree_style(clade_tree)
            clade_tree.render(
                f"{output_dir}{clade.replace(' ', '_')}_w{'' if color_by_cat else 'o'}_color_by_order.pdf",
                tree_style=ts,
                h=120,
                w=120,
            )
    else:
        ts = get_tree_style(tree)
        tree.render(
            f"{output_dir}complete_angiosperms_tree_w{'' if color_by_cat else 'o'}_color_by_clade.pdf",
            tree_style=ts,
            h=120,
            w=120,
        )


if __name__ == "__main__":
    draw_tree()
