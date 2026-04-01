import logging
from dask.delayed import Delayed
import numpy as np
import plotly.graph_objects as go
import dask.array as da


import organelle_morphology
from organelle_morphology.util import (
    bounding_box_delayed,
)


def organelle_types() -> list[str]:
    """The list of organelles currently implemented.

    The strings used here to encode the organelles are expected in
    various APIs when referring to a specific organelle.
    """
    return list(organelle_registry.keys())


class Organelle:
    _name = "organelle_name"
    logger = logging.getLogger(__name__)

    def __init__(self, source: "organelle_morphology.DataSource", label: int):
        """The organelle base class

        Holds references to its mesh and label.
        Statistical data is managed centrally by the Project properties.

        Note that instances of Organelle typically are not instantiated directly,
        but through the corresponding subclass of OrganelleFactory.

        Args:
            source: The source object containing this organelle
            label: label used in the original data for this organelle.
        """
        self.source = source
        self.label = label
        self._organelle_id = f"{self._name}_{str(label).zfill(4)}"

        # Keep heavy 3D objects locally
        self._skeleton = None
        self._sampled_skeleton = None

    @classmethod
    def construct(cls, source, labels: list[int]):
        """A trivial factory method for organelle instances.

        It constructs an instance per label. The construction process for each
        organelle is independent of all others. Other organelles can subclass
        this to implement a construction process that e.g. takes into account
        all organelle instances.
        """
        for label in labels:
            yield organelle_registry[cls._name](
                source=source,
                label=label,
            )

    def __repr__(self):
        return f"{self.__class__.__name__}({self._organelle_id})"

    def plotly_skeleton(self):
        if self._skeleton is None:
            return None

        nodes = self.skeleton.vertices
        edges = self.skeleton.edges

        line_width = 10

        # Create a 3D line plot for the edges
        x_values = []
        y_values = []
        z_values = []

        for edge in edges:
            x_values.extend(
                [nodes[edge[0]][0], nodes[edge[1]][0], None]
            )  # add None to separate lines
            y_values.extend(
                [nodes[edge[0]][1], nodes[edge[1]][1], None]
            )  # add None to separate lines
            z_values.extend(
                [nodes[edge[0]][2], nodes[edge[1]][2], None]
            )  # add None to separate lines

        skeleton_trace = go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line=dict(width=line_width),  # Set line width
            name=f"Skeleton_{self.id}",  # Set label
        )
        return skeleton_trace

    def plotly_mesh(
        self,
        show_curvature: bool = False,
        show_skeleton: bool = False,
        mcs_label=False,
        mcs_filter_ids=None,
    ):
        # prepare the plotly mesh object for visualization

        verts = self.mesh.compute().vertices
        faces = self.mesh.compute().faces

        # prepare data for plotly
        vertsT = np.transpose(verts)
        facesT = np.transpose(faces)

        # initialize basic drawing settings
        intensity = None
        colorscale = None
        opacity = 1

        # override settings if special visualization is requested
        if show_curvature:
            curvature_vertices = self.curvature_map
            intensity = curvature_vertices
            colorscale = "Viridis"
            opacity = 1

        if show_skeleton:
            opacity = 0.7

        # add coloration for the close regions
        if mcs_label:
            self.logger.warning(
                "MCS visualization is temporarily disabled during SoA refactor."
            )

        go_mesh = go.Mesh3d(
            x=vertsT[0],
            y=vertsT[1],
            z=vertsT[2],
            i=facesT[0],
            j=facesT[1],
            k=facesT[2],
            name=self.id,
            opacity=opacity,
            intensity=intensity,
            colorscale=colorscale,
            showscale=False,
        )
        return go_mesh

    @property
    def skeleton(self):
        """Get the skeleton for this organelle"""

        return self._skeleton

    @skeleton.setter
    def skeleton(self, value):
        self._skeleton = value

    @property
    def skeleton_info(self):
        """Get the skeleton info for this organelle from the central properties DataFrame."""
        if not self._skeleton:
            return None
        df = self.source.project.properties._prop_df
        if self.id in df.index:
            return df.loc[self.id].to_dict()
        return {}

    @property
    def sampled_skeleton(self):
        """Get the sampled skeleton for this organelle.
        This included the sampled points,
        as well as the corresponding reference point to later get the correct plane normal vector
        """
        if self._skeleton is None:
            self.logger.warning(
                f"Skeleton has not been generated for {self.id} yet. Please run project.generate_skeletons() first."
            )
            return None

        return self._sampled_skeleton

    @sampled_skeleton.setter
    def sampled_skeleton(self, value):
        self._sampled_skeleton = value

    @property
    def mesh(self) -> Delayed:
        """Get the mesh for this organelle"""
        return self.source.meshes[self.label]

    def get_mesh_mcs_colored(self, mcs_label=None) -> Delayed:
        """Get mcs colored delayed meshes"""
        self.logger.warning(
            "MCS visualization is temporarily disabled during SoA refactor."
        )
        return self.mesh

    @property
    def id(self):
        """Get the organelle ID of this organelle"""

        return self._organelle_id

    @property
    def bounding_box(self) -> Delayed:
        return bounding_box_delayed(self.mesh).compute()

    @property
    def geometric_data(self):
        """Get the geometric data for this organelle from the central properties DataFrame."""
        df = self.source.project.properties._prop_df
        if self.id in df.index:
            return df.loc[self.id].to_dict()
        return {}

    @property
    def mesh_properties(self):
        """Get the mesh data for this organelle from the central properties DataFrame."""
        df = self.source.project.properties._prop_df
        if self.id in df.index:
            return df.loc[self.id].to_dict()
        return {}

    @property
    def curvature_map(self) -> np.ndarray:
        """Get the mesh data for this organelle"""
        return self.source.calc_curvature(self.label)[self.label]

    @property
    def curvature_mesh(self) -> Delayed:
        return self.source.get_meshes_curvature_colored(labels=self.label)[0]

    @property
    def data(self) -> da.Array:
        """Get the raw data for this organelle
        by filtering the data of the source object"""
        source_ds = self.source.data[:]
        return da.where(source_ds == self.label, source_ds, 0)


class Mitochondrium(Organelle):
    _name = "mito"


class EndoplasmicReticulum(Organelle):
    _name = "er"


class AutoFillDict(dict):
    """Dictionary that populates itself"""

    def __missing__(self, key: str):
        new = type(key.capitalize(), (Organelle,), {"_name": key})
        self[key] = new
        return new


# The dictionary of registered organelle subclasses, mapping names
# to classes
organelle_registry: dict[str, "Organelle"] = AutoFillDict()
organelle_registry["mito"] = Mitochondrium
organelle_registry["er"] = EndoplasmicReticulum
