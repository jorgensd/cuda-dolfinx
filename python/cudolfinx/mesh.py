# Copyright (C) 2026 Benjamin Pachev
#
# This file is part of cuDOLFINX
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Functions to create distributed meshes on GPUs."""

from cudolfinx import cpp as _cucpp
from dolfinx import mesh


def ghost_layer_mesh(domain: mesh.Mesh):
    """Add a ghost layer of cells to the given mesh."""
    geometry = domain._geometry._cpp_object
    if hasattr(geometry, "cmaps"):
        # DOLFINx >= 0.11 supports multiple coordinate elements
        if len(geometry.cmaps) != 1:
            raise NotImplementedError("Meshes with multiple cell types are not supported.")
        cmap = geometry.cmaps[0]
    else:
        cmap = geometry.cmap
    _ghost_mesh = _cucpp.fem.ghost_layer_mesh(domain._cpp_object, cmap)
    return mesh.Mesh(
        _ghost_mesh,
        domain._ufl_domain
    )

def ghost_layer_meshtags(meshtags: mesh.MeshTags, ghosted_mesh: mesh.Mesh):
    """Transfer meshtags to ghost layer mesh."""
    _cpp_meshtags = _cucpp.fem.ghost_layer_meshtags(
        meshtags._cpp_object,
        ghosted_mesh.topology._cpp_object
    )
    return mesh.MeshTags(_cpp_meshtags)
