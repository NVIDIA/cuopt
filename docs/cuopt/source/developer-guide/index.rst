..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

==========================
Developer Guide
==========================

This section is for people working **on** cuOpt rather than with it. It documents
how the project is put together: what runs where, what depends on what, and which
build artifact holds which code.

If you are using cuOpt to solve problems, you do not need anything here. Start with
:doc:`../introduction` and the API section for your language.

:doc:`library-architecture` is the starting point. It covers the run-time topology
(client machine versus GPU host), the dependency edges between the Python packages
and the C++ libraries, the build artifacts those come from, and the failure modes
that recur when code moves between translation units, libraries, or packages.

Contributors working in the C++ sources should also read ``cpp/docs/DEVELOPER_GUIDE.md``
and ``cpp/docs/grpc-server-architecture.md`` in the repository, which go deeper into
C++ conventions and the gRPC server's internals than the published documentation does.

.. toctree::
   :maxdepth: 2
   :caption: In this section
   :name: developer-guide-contents

   library-architecture.md
