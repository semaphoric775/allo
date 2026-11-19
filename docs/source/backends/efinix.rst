..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

######################################
Getting Started on Efinix FPGAs
######################################

Efinix Titanium and Trion FPGAs are programmed with the Efinity toolchain, which only accepts
Verilog, SystemVerilog, or VHDL. Allo targets them through its :doc:`XLS backend <xls>`: designs
are compiled to plain Verilog with no vendor primitives or IP, simulated cycle-accurately with
Verilator, and packaged as an Efinity project for synthesis and place and route.

.. code-block:: text

   Allo kernel ──► XLS[cc] C++ ──► xlscc ──► XLS IR ──► opt_main ──► codegen_main ──► Verilog
                                                                                        │
                        Verilator cosim  ◄──────────────────────────────────────────────┤
                        Efinity (map / place & route / bitstream)  ◄────────────────────┘

.. contents:: Table of Contents
   :local:
   :depth: 2


Prerequisites
=============

Besides a working Allo installation (see :ref:`setup`), you need three tools. Only Efinity is
needed for synthesis; XLS and Verilator are enough to generate and simulate RTL.

Google XLS
----------

The prebuilt XLS releases include ``opt_main`` and ``codegen_main`` but not ``xlscc``, the C++
frontend, which has to be built from source with Bazel. Use the Bazel version pinned in the
repository's ``.bazelversion`` (currently 8.7.0; Bazel 9 fails to analyse the dependencies).

.. code-block:: bash

   # 1. Prebuilt tools (pick a release from https://github.com/google/xls/releases)
   mkdir -p ~/tools/xls && cd ~/tools/xls
   curl -LO https://github.com/google/xls/releases/download/<tag>/xls-<tag>-linux-x64.tar.gz
   tar xzf xls-<tag>-linux-x64.tar.gz

   # 2. xlscc from the same commit (about 20 minutes on 10 cores)
   git clone https://github.com/google/xls.git src && cd src
   git checkout <commit of the release>
   bazel build -c opt //xls/contrib/xlscc:xlscc

   # 3. One directory with every tool, and the headers xlscc needs
   mkdir -p ~/tools/xls/bin ~/tools/xls/include
   ln -sf ~/tools/xls/xls-<tag>-linux-x64/*_main ~/tools/xls/bin/
   ln -sf ~/tools/xls/src/bazel-bin/xls/contrib/xlscc/xlscc ~/tools/xls/bin/
   ln -sfn "$(bazel info output_base)/external/+ac_types_extension+ac_datatypes" \
       ~/tools/xls/include/ac_datatypes

The XLS headers include ``ac_datatypes/include/ac_int.h``, but Bazel names the external
directory ``+ac_types_extension+ac_datatypes``, hence the symlink.

Verilator
---------

Verilator 5 or newer is used for co-simulation (``apt install verilator`` or a source build).

Efinix Efinity
--------------

Install Efinity from the Efinix support center. Allo only needs its location; it sources
``bin/setup.sh`` in a subshell for every Efinity call.

.. note::

   Do not ``source`` Efinity's ``setup.sh`` in the shell you run Allo from: it puts Efinity's
   bundled Python first on ``PATH`` and hides your Allo environment.

Environment
-----------

.. code-block:: bash

   export XLS_BIN=~/tools/xls/bin
   export XLS_INCLUDE=~/tools/xls/src/xls/contrib/xlscc/synth_only:~/tools/xls/include
   export EFINITY_HOME=~/efinity/<version>

Check the setup with the XLS backend tests; they are skipped when a tool is missing:

.. code-block:: bash

   python3 -m pytest tests/test_xls.py -v -rs


Your First Accelerator
======================

Write and simulate
------------------

A vector addition with 8-bit inputs, stored in block RAM:

.. code-block:: python

   import numpy as np
   import allo
   from allo.ir.types import int8

   def vadd(a: int8[64], b: int8[64]) -> int8[64]:
       c: int8[64] = 0
       for i in allo.grid(64):
           c[i] = a[i] + b[i]
       return c

   s = allo.customize(vadd)
   mod = s.build(target="xls", mode="cosim", project="vadd.prj", use_memory=True)

   a = np.random.randint(-64, 64, 64).astype(np.int8)
   b = np.random.randint(-64, 64, 64).astype(np.int8)
   np.testing.assert_array_equal(mod(a, b), a + b)  # runs the RTL in Verilator
   print(mod.cycles, mod.schedule)

``mode="cosim"`` generates the RTL and a Verilator testbench; calling the module simulates the
hardware. Use ``mode="rtl"`` when you only need the Verilog. ``vadd.prj`` contains:

- ``vadd_core.v``: the pipeline generated by XLS.
- ``vadd.v``: the top-level module, with ready/valid ports ``v0_in``, ``v1_in`` (inputs) and
  ``out`` (output), each with ``_vld`` / ``_rdy`` signals, plus ``clk`` and an active-high
  synchronous ``rst``. It instantiates one RAM per memory.
- ``allo_ram.v``: block-RAM models written so that Efinity infers ``EFX_RAM10`` (Titanium) or
  ``EFX_RAM_5K`` (Trion) blocks.
- ``vadd_axis.v``: an AXI4-Stream wrapper (see `Integrating with a System`_).

Synthesize, place and route
---------------------------

.. code-block:: python

   report = mod.export_efinity(family="Titanium", device="Ti60F225", clock_mhz=200, flow="pnr")
   print(report)  # {'FFs': ..., 'LUTs': ..., 'RAMs': ..., 'DSP/MULTs': ..., 'fmax_mhz': ...}

``export_efinity`` writes an Efinity project (``vadd.prj/efinity/vadd.xml`` with a ``clk``
constraint in ``vadd.sdc``) that can also be opened in the Efinity GUI. ``flow="map"`` stops after
synthesis; ``flow="pnr"`` also places and routes and reports the maximum frequency. For Trion,
pass e.g. ``family="Trion", device="T20F256"``. ``timing_model`` (default ``"C4"``) selects the
speed grade.

.. note::

   The generated project leaves the I/O assignment to Efinity, which is enough for resource and
   timing estimates. To build a bitstream for a board, assign pins and clocks in Efinity's
   Interface Designer, or instantiate the accelerator inside your own design (see below).


Using the FPGA Resources
========================

Block RAM
---------

With ``use_memory=True`` every array becomes a memory backed by a block RAM; scalars stay in
registers. Without it, arrays are kept in registers, which suits small arrays.
``s.bind_storage`` / ``Memory`` annotations select single-port (``RAM_1P``) or simple dual-port
(``RAM_2P``, default) memories.

Partitioning
------------

A memory provides one access per port per cycle. ``s.partition`` splits an array into several
memories (banks):

.. code-block:: python

   from allo.customize import Partition

   s = allo.customize(vadd)
   s.partition(s.a, Partition.Cyclic, dim=1, factor=4)  # 4 RAMs: a_b0 .. a_b3
   s.partition(s.b, Partition.Cyclic, dim=1, factor=4)
   s.partition(s.c, Partition.Cyclic, dim=1, factor=4)
   s.unroll("i", factor=4)

``Partition.Block`` splits an array into contiguous blocks; ``Partition.Complete`` on a dimension
creates one bank per index (a bank of a single element is kept in registers).

.. note::

   Partitioning is functionally complete, but the current XLS scheduling does not yet turn banks
   into extra parallelism: in our measurements a cyclically partitioned and unrolled GEMM ran no
   faster than the unpartitioned one, and slightly slower unless xlscc is passed
   ``--split_states_on_channel_ops=false`` (via ``configs={"xlscc_flags": [...]}``). Measure with
   ``mod.cycles`` before relying on it. For designs dominated by streaming I/O, which moves one
   word per cycle, keeping small arrays in registers (``use_memory=False``) is often faster than
   block RAM.

DSP blocks
----------

Multiplications are mapped by Efinity to the hard multipliers (DSP blocks on Titanium, multipliers
on Trion) without annotations.

Fixed point
-----------

Efinix fabrics have no floating-point hard blocks, and XLS does not support floating point, so
use ``Fixed`` / ``UFixed``. Values travel as raw fixed-point bits; in cosim, real-valued NumPy
inputs are converted for you and results are converted back, bit-exact with ``s.build()``:

.. code-block:: python

   from allo.ir.types import Fixed

   F = Fixed(16, 8)

   def fmac(a: F[8], b: F[8]) -> F[8]:
       c: F[8] = 0
       for i in range(8):
           c[i] = a[i] * b[i] + a[i]
       return c

   s = allo.customize(fmac)
   mod = s.build(target="xls", mode="cosim", project="fmac.prj")
   a = np.random.randint(-64, 64, 8) / 16.0
   b = np.random.randint(-64, 64, 8) / 16.0
   np.testing.assert_array_equal(mod(a, b), s.build()(a, b))


Dataflow Designs
================

Multi-kernel ``@df.region`` designs compile each kernel into its own module, so kernels run
concurrently, and connect streams with FIFOs of the declared depth:

.. code-block:: python

   import allo.dataflow as df
   from allo.ir.types import int32, Stream

   N = 16

   @df.region()
   def top(A: int32[N], B: int32[N]):
       pipe: Stream[int32, 4]

       @df.kernel(mapping=[1], args=[A])
       def producer(local_A: int32[N]):
           for i in range(N):
               pipe.put(local_A[i] + 1)

       @df.kernel(mapping=[1], args=[B])
       def consumer(local_B: int32[N]):
           for i in range(N):
               local_B[i] = pipe.get() * 2

   mod = df.build(top, target="xls", mode="cosim", project="df.prj")
   A = np.arange(N, dtype=np.int32)
   B = np.zeros(N, dtype=np.int32)
   mod(A, B)  # B is written by the simulated hardware
   print(mod.export_efinity(family="Trion", device="T20F256", flow="pnr"))

Region arguments that a kernel reads arrive on ``v<k>_in`` and arguments it writes leave on
``v<k>_out`` (``k`` is the argument position). Arguments shared by several kernels are broadcast;
arguments written by several kernels (e.g. each PE of a tiled GEMM updating its own tile) are
merged element by element. ``configs={"use_memory": True}`` puts kernel arrays in block RAM, and
``fifo_reg_bits`` (default 1024) sets the size up to which stream FIFOs use registers instead of
block RAM. See :doc:`xls` for the details.


Integrating with a System
=========================

Every streaming design (and every dataflow region) also gets ``<top>_axis.v``, an AXI4-Stream
wrapper for DMA engines or an SoC such as the Efinix Sapphire RISC-V SoC:

- inputs ``s_axis_v<k>_in_tdata`` / ``_tvalid`` / ``_tready``,
- outputs ``m_axis_<name>_tdata`` / ``_tvalid`` / ``_tready`` / ``_tlast``, with ``tlast`` on the
  last word of each result array,
- clock ``aclk`` and active-low reset ``aresetn``.

Arrays are transferred one element per beat in row-major order. Use it as the project top with
``export_efinity(..., interface="axis")``, or add the ``.v`` files from the project directory to
your own Efinity project.


Tuning Performance
==================

Pass options through ``configs``:

.. code-block:: python

   mod = s.build(target="xls", mode="rtl", project="p.prj", use_memory=True,
                 configs={"clock_period_ps": 12})

- ``clock_period_ps`` (default 8): how much logic XLS may put in one pipeline stage. XLS has no
  Efinix delay model, so it uses a unit model in which this is roughly the number of operations
  per stage; lower values pipeline more deeply. Check the result with ``flow="pnr"``.
- ``pipeline_stages``: fix the number of stages instead.
- ``xlscc_flags``: extra command-line flags for xlscc, e.g.
  ``["--split_states_on_channel_ops=false"]``.
- By default the flow prefers full throughput and relaxes ``clock_period_ps`` (up to 4x) before
  accepting a lower initiation rate; ``worst_case_throughput`` forces a rate.
  ``mod.schedule`` records what was used.
- ``mod.cycles`` (cosim) reports the cycle count of a run.


Troubleshooting
===============

``XLS tool 'xlscc' not found``
   Set ``XLS_BIN`` to the directory containing ``xlscc``, ``opt_main`` and ``codegen_main``.

``'ac_datatypes/include/ac_int.h' file not found``
   ``XLS_INCLUDE`` must contain ``xls/contrib/xlscc/synth_only`` and a directory holding the
   ``ac_datatypes`` symlink described above.

Tests or imports fail after enabling Efinity
   The Python on ``PATH`` was replaced by Efinity's; open a new shell, and only set
   ``EFINITY_HOME``.

``Floating-point type ... is not supported``
   Switch to ``Fixed`` / ``UFixed`` types.

``Efinity map failed``
   See ``<project>/efinity/outflow/<top>.err.log``.
