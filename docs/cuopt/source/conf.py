# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from cuopt_server.webserver import app as cuoptapp
from fastapi.openapi.utils import get_openapi
import datetime
import cuopt
import yaml
from packaging.version import Version
import subprocess
import sys
import os
# Run cuopt server help command and save output
subprocess.run(["python", "-m", "cuopt_server.cuopt_service", "--help"],
              stdout=open("cuopt-server/server-api/server-cli-help.txt", "w"))
# Run cuopt_sh help command and save output
subprocess.run(["cuopt_sh", "--help"], stdout=open("cuopt-server/client-api/sh-cli-help.txt", "w"))

# Run cuopt_cli help command and save output
subprocess.run(["cuopt_cli", "--help"], stdout=open("cuopt-cli/cuopt-cli-help.txt", "w"))

with open('cuopt_spec.yaml', 'w') as f:
    yaml.dump(get_openapi(
        title=cuoptapp.title,
        version=cuoptapp.version,
        openapi_version=cuoptapp.openapi_version,
        description=cuoptapp.description,
        summary=cuoptapp.summary,
        routes=cuoptapp.routes,
    ), f)

# -- General configuration ------------------------------------------------

sys.path.insert(0, os.path.abspath("cpp/"))

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx.domains.cpp",
    "sphinx.domains.c",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx_design",
    "sphinx_markdown_tables",
    "sphinx.ext.doctest",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "swagger_plugin_for_sphinx",
]

swagger = [
    {
        "name": "cuOpt API",
        "id": "cuopt-api",
        "page": "cuopt-api",
        "options": {
            "url": "_static/cuopt_spec.yaml",
            "deepLinking": "true",
            "showExtensions": "true",
            "defaultModelsExpandDepth" : -1
        }
    },
]

nbsphinx_execute = 'never'
ipython_mplbackend = 'str'

# Add any files to exclude from the build
exclude_patterns = ["hidden"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "NVIDIA cuOpt"
copyright = f"2021-{datetime.datetime.today().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# -- Options for HTML output ----------------------------------------------
# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
CUOPT_VERSION = Version(cuopt.__version__)
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# The short X.Y version.
version = f"{CUOPT_VERSION.major:02}.{CUOPT_VERSION.minor:02}"
# The full version.
release = (
    f"{CUOPT_VERSION.major:02}.{CUOPT_VERSION.minor:02}.{CUOPT_VERSION.micro:02}"
)

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nvidia_sphinx_theme"
html_logo = 'images/main_nv_logo_square.png'
html_title = f'{project} ({version})'

autosectionlabel_prefix_document = True

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 6,
     "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
    'extra_head': [  # Adding Adobe Analytics
        '''
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    '''
    ],
    'extra_footer': [
        '''
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    '''
    ],
    "show_nav_level": 2
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["swagger-nvidia.css"]
html_extra_path = ["project.json", "versions1.json"]


# -- Options for Breathe (Doxygen) ----------------------------------------
breathe_projects = {
    "libcuopt": "../../../cpp/doxygen/xml"
}
breathe_default_project = "libcuopt"

# Configure Breathe to handle file types
breathe_domain_by_extension = {
    "hpp": "cpp",
    "h": "c",
    "c": "c"
}
breathe_implementation_filename_extensions = ['.cpp', '.cu']

# Configure Breathe to handle source files
breathe_projects_source = {
    "libcuopt": ("../../../cpp/src", [
        "*.hpp",
        "*.h",
        "*.cuh",
        "*.cu"
    ])
}

# Configure Breathe to handle CUDA and template attributes
breathe_doxygen_aliases = {
    "int32_t": "int32_t",
    "int8_t": "int8_t",
}

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "cuopt.tex",
        "cuopt Documentation",
        "NVIDIA Corporation",
        "manual",
    )
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "cuopt", "cuopt Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "cuopt",
        "cuopt Documentation",
        author,
        "cuopt",
        "One line description of project.",
        "Miscellaneous",
    )
]

# Config numpydoc
numpydoc_show_inherited_class_members = {
    # option_context inherits undocumented members from the parent class
    "cuopt.option_context": False,
}

# Rely on toctrees generated from autosummary on each of the pages we define
# rather than the autosummaries on the numpydoc auto-generated class pages.
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False

autoclass_content = "class"

autodoc_mock_imports = [
    "cuopt.distance_engine.waypoint_matrix_wrapper.WaypointMatrix",
    "cuopt.routing.vehicle_routing_wrapper.DataModel",
    "cuopt.routing.vehicle_routing_wrapper.SolverSettings",
    "cuopt.routing.DataModel.add_order_precedence",
]

nitpick_ignore = [
    ("py:class", "cuopt.routing.vehicle_routing_wrapper.SolverSettings"),
    ("py:class", "cuopt.routing.vehicle_routing_wrapper.DataModel"),
    ("py:class", "cuopt.distance_engine.waypoint_matrix_wrapper.WaypointMatrix"),
    ("py:class", "enum.Enum"),
    ("py:obj",   "cuopt.routing.DataModel.add_order_precedence"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.denominator"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.imag"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.numerator"), 
    ("py:obj",   "cuopt_sh_client.SolverMethod.real"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.as_integer_ratio"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.bit_count"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.bit_length"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.conjugate"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.from_bytes"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.is_integer"),
    ("py:obj",   "cuopt_sh_client.SolverMethod.to_bytes"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.as_integer_ratio"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.conjugate"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.from_bytes"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.to_bytes"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.is_integer"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.bit_count"),
    ("py:obj",   "cuopt_sh_client.PDLPSolverMode.bit_length"),
    ("c:type", "size_t"),
    ("c:identifier", "int32_t"),
    ("c:identifier", "int8_t"),
]

def skip_unwanted_inherited_members(app, what, name, obj, skip, options):
    inherited_to_skip = {"as_integer_ratio", "conjugate", "from_bytes", "to_bytes", "is_integer", "bit_count", "bit_length", "is_integer"}  # add more as needed
    if name in inherited_to_skip:
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_unwanted_inherited_members)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Search settings
html_search_language = 'en'
html_search_options = {
    'type': 'default'
}
html_search = True

def setup(app):
    from sphinx.application import Sphinx
    from typing import Any, List

    app.setup_extension("sphinx.ext.autodoc")
    app.connect("autodoc-skip-member", skip_unwanted_inherited_members)

