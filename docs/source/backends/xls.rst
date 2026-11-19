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

############################
Google XLS (ASIC)
############################

The `Google XLS <https://google.github.io/xls/>`_ ASIC backend leverages the Google Accelerated Hardware Synthesis toolchain to generate 
hardware accelerators for the ASIC flow. This document demonstrates how to define a general matrix multiplication (GEMM) 
kernel using the Allo ADL and generate HLS code for place and route. For more details on kernel customizations and scheduling optimizations, 
please refer to the `Allo-HLS tutorial <https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html>`_.

Kernel Definition
-----------------
The GEMM kernel is implemented with `int32` precision using pre-defined constants for the matrix dimensions. The kernel utilizes the 
`allo.grid` API to iterate over output indices and the `allo.reduction` API to designate the reduction axis for accumulating the 
dot-product.

.. code-block:: python

   import allo
   from allo.ir.types import int32
   import numpy as np

   # Define matrix dimensions
   M, N, K = 4, 4, 4

   def gemm(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
       C: int32[4, 4] = 0
       for i, j, k in allo.grid(4, 4, 4):
           C[i, j] += A[i, k] * B[k, j]
       return C

Code Generation for Google XLS
------------------------------------
Allo supports several approaches to generate HLS code:

1. **Direct HLS Code Generation**:  
   Set the target to `"xlscc"` to produce XLS C++ frontend code as a string. This code includes the necessary headers and pragmas for 
   Google XLS synthesis. We expose a `use_memory` flag to control whether to use the memory API in the XLS backend.

    .. code-block:: python

       code = s.build(target="xlscc", use_memory=True)
       print(code)

   When `use_memory` is set to `True`, the generated code will use the memory API in the XLS backend. When `use_memory` is set to `False`, 
   the generated code will use the C-style arrays which become registers clusters in the XLS backend.

2. **HLS Emulation, Synthesis, and Execution**:  
   Users may also specify the target as `"xlscc"` but run the designs in our software emulation mode:

   - ``sw_emu``: Software emulation mode, which is similar to C simulation that compiles the program using C compiler 
   and runs it on the CPU. Depending on the size of your input data, this mode may take within one minute.

    .. code-block:: python

       mod = s.build(target="xlscc", mode="sw_emu", project="gemm.prj")

   - ``rtl``: Runs ``xlscc`` → ``opt_main`` → ``codegen_main`` and emits vendor-neutral
     Verilog (no FPGA primitives or IP), suitable for any synthesis tool (e.g., Efinix Efinity, Yosys).
     The project contains ``<top>_core.v`` (XLS pipeline), ``<top>.v`` (top wrapper that instantiates
     a RAM model for each external memory), and ``allo_ram.v`` (inferable block-RAM models).
   - ``cosim``: Same as ``rtl``, then builds a Verilator testbench; calling the module simulates the
     RTL cycle-accurately and ``mod.cycles`` holds the cycle count.

    .. code-block:: python

       mod = s.build(target="xlscc", mode="cosim", project="gemm.prj", use_memory=True,
                     configs={"clock_period_ps": 8})
       np.testing.assert_array_equal(mod(A, B), A @ B)

   XLS binaries are looked up in ``$XLS_BIN`` then ``PATH``; xlscc header directories are taken from
   ``configs["include_dirs"]`` or ``$XLS_INCLUDE`` (colon separated). They must contain
   ``xls/contrib/xlscc/synth_only`` and a directory holding ``ac_datatypes/include/ac_int.h``
   (in a Bazel build, ``external/+ac_types_extension+ac_datatypes``, which needs to be symlinked
   as ``ac_datatypes``). ``worst_case_throughput`` sets how many cycles may separate proc
   activations. By default (``throughput_first=True``) the flow first relaxes
   ``clock_period_ps`` (up to 4x) to reach full throughput, since the unit delay model is
   pessimistic for FPGA fabric, and only then accepts the reduced throughput suggested by the
   XLS scheduler. The schedule actually used is stored in ``mod.schedule``. Other ``configs`` keys: ``delay_model``
   (default ``"unit"``, as XLS has no FPGA delay model, so ``clock_period_ps`` bounds the ops per
   stage), ``pipeline_stages`` (overrides ``clock_period_ps``), ``reset`` (default ``"rst"``,
   active-high synchronous), ``use_system_verilog`` (default ``False``), ``xlscc_flags`` (extra
   xlscc flags) and ``max_cycles``.
   Streaming designs expose ``v<k>_in`` / ``out`` ready/valid ports; scalar designs expose their
   arguments plus ``in_vld`` / ``out_vld``.

   The generated RTL can be packaged as an Efinix Efinity project (written to ``<project>/efinity``)
   and optionally synthesized (``flow="map"``) or placed and routed (``flow="pnr"``), which returns
   the resource estimate and, after place and route, ``fmax_mhz``. Set ``EFINITY_HOME`` to the
   Efinity install directory (Efinity's ``setup.sh`` is sourced in a subshell, so it does not
   replace the Python environment). The project is derived from an example shipped with the
   installed Efinity release so that it matches its project format.

    .. code-block:: python

       mod = s.build(target="xlscc", mode="rtl", project="vadd.prj", use_memory=True)
       print(mod.export_efinity(family="Titanium", device="Ti60F225", clock_mhz=200, flow="pnr"))
       # {'FFs': 269, 'LUTs': 281, 'RAMs': 4, ..., 'fmax_mhz': 320.7}

   Streaming designs also get ``<top>_axis.v``, an AXI4-Stream wrapper (``s_axis_v<k>_in_*`` inputs,
   ``m_axis_out_*`` output with ``tlast`` on the last word of each result, active-low ``aresetn``)
   for connecting to DMA engines or an SoC such as Efinix Sapphire. Pass ``interface="axis"`` to
   ``export_efinity`` to use it as the top-level module. Supported families are ``"Titanium"``
   (e.g., ``Ti60F225``) and ``"Trion"`` (e.g., ``T20F256``).

   **Data types.** Integer types of any width are supported; channels carry the element type
   (e.g., 8-bit ports for ``int8``) and cosim results are sign-extended accordingly. Fixed-point
   types (``Fixed`` / ``UFixed``) are lowered to raw integers in ``rtl`` / ``cosim`` mode, exactly
   as in the LLVM backend, so results match ``s.build()`` bit for bit; in cosim, real-valued
   inputs are converted to raw fixed-point bits and outputs back to real values. Floating point is
   not supported by XLS (and Efinix fabrics have no floating-point hard blocks); use fixed point.

3. **Memories and Partitioning** (``use_memory=True``):
   Every array becomes an XLS memory backed by a block RAM. ``s.partition`` splits an array into
   banks: ``Partition.Cyclic`` / ``Partition.Block`` with ``factor=F`` produce ``F`` memories
   (``<name>_b<k>``); a partition that leaves one element per bank keeps the array in registers.
   (The current XLS scheduling does not yet exploit banks for extra parallelism; see
   :doc:`efinix`.) Scalars always stay in registers.
   XLS cannot see dependences through external RAMs, so a loop that reads and writes the same
   element at a loop-invariant address (e.g., the accumulator ``C[i, j]`` of a reduction over
   ``k``) keeps that element in a register for the duration of the loop.

4. **Dataflow Regions**:
   ``df.build(top, target="xls", mode="rtl" | "cosim", project=...)`` compiles every kernel of a
   ``@df.region`` into its own XLS module, so kernels run concurrently, and generates a Verilog top
   that connects each ``Stream`` through a FIFO of the declared depth. Region array arguments read
   by a kernel are streamed in on ``v<k>_in``; arguments written by a kernel are streamed out on
   ``v<k>_out`` (row-major). In ``cosim`` mode, calling the module fills the output arguments in
   place. ``configs`` accepts ``use_memory`` and ``fifo_reg_bits`` (FIFOs up to this size, default
   1024 bits, are kept in registers; larger ones use block RAM) in addition to the options above.
   A region argument read by several kernels is broadcast to all of them. When several kernels
   write the same argument (e.g., each PE of a tiled GEMM updates its own tile of ``C``), every
   writer streams the whole array together with a per-element "written" bit and the outputs are
   merged element by element; elements no writer touched are 0. With a single writer, elements it
   does not write are only preserved if the kernel also reads the argument. Streams must have one
   producer and one consumer. The returned module also provides ``export_efinity``.

    .. code-block:: python

       mod = df.build(top, target="xls", mode="cosim", project="systolic.prj")
       mod(A, B, C)  # C is written by the RTL simulation
       print(mod.export_efinity(family="Trion", device="T20F256", flow="pnr"))

Project Structure and Execution
-------------------------------
The generated XLS project (e.g., in the folder ``gemm.prj``) typically includes:

- **test_block.cpp**: The accelerator kernel code. 

If you set `use_memory` to `True`, the generated project will also include:

- **rewrites.textproto**: The memory configuration file that is required by the XLS 
backend to map any memory arrays to the hardware. Without this file or if this file is
configured wrong, the XLS backend will error out during its IR optimization pass.

If you are using the software emulation mode, you will also obtain the following files:

- **test_harness.cpp**: The host-side (CPU) code that invokes the accelerator.
- **Makefile**: Build scripts to compile the project.
- **input*.data**: The binary input data files for the design.

To run the design, prepare the input matrices using NumPy and invoke the generated module to obtain the result:

.. code-block:: python

   np_A = np.random.random((M, K)).astype(np.int32)
   np_B = np.random.random((K, N)).astype(np.int32)
   result = mod(np_A, np_B)
   np.testing.assert_allclose(result, np.matmul(np_A, np_B), rtol=1e-5, atol=1e-5)
   
Note:
  The XLS backend only supports integer and fixed-point types. Floating-point types (f16, f32, f64) are not supported.

Memory Implementation Customization
------------------------------------
Allo provides the ``Memory`` class to specify on-chip memory implementation details for arrays, similar to the `Vitis HLS bind_storage pragma <https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-bind_storage>`_. This allows fine-grained control over how arrays are mapped to FPGA memory resources (BRAM, URAM, LUTRAM, etc.).

**Basic Usage**

Use the ``@`` operator to annotate function arguments or local variables with memory specifications:

.. code-block:: python

   from allo import Memory
   from allo.ir.types import int32, float32

   # Define memory specifications
   MemUram = Memory(resource="URAM", storage_type="RAM_1P")
   MemBram = Memory(resource="BRAM", storage_type="RAM_2P")

   def kernel(a: int32[32] @ MemUram, b: int32[16, 16] @ MemBram) -> int32[32]:
       # Local variable with memory annotation
       buf: int32[32] @ Memory(resource="BRAM")
       for i in range(32):
           buf[i] = a[i] * 2
       c: int32[32]
       for i in range(32):
           c[i] = buf[i] + 1
       return c

   s = allo.customize(kernel)
   mod = s.build(target="xls", use_memory=True)
   print(mod.hls_code)

This generates a rewrite textproto file that maps buffer ``a`` (``RAM_1P``) to the XLS ``RAM_1RW`` type and input buffer ``b`` (``RAM_2P``) to the XLS ``RAM_1R1W`` type:

**Memory Class Parameters**
The Allo ``Memory`` class accepts the following parameters:

- **resource** (str): Memory resource type

  - ``"BRAM"``: Block RAM - the most common on-chip memory
  - ``"URAM"``: Ultra RAM - larger capacity, available on UltraScale+ devices
  - ``"LUTRAM"``: LUT-based RAM - faster but smaller
  - ``"SRL"``: Shift Register LUT - efficient for FIFOs
  - ``"AUTO"``: Let the HLS tool decide (default)

These all default to RAM_ABSTRACT in XLS.

- **storage_type** (str, optional): RAM access pattern

  - ``"RAM_1P"``: Single-port RAM
  - ``"RAM_2P"``: Simple dual-port RAM (one read, one write port)
  - ``"RAM_T2P"``: True dual-port RAM (two read/write ports)
  - ``"RAM_1WNR"``: Single write, N read ports
  - ``"RAM_S2P"``: Simple dual-port (alias for RAM_2P)
  - ``"ROM_1P"``: Single-port ROM
  - ``"ROM_2P"``: Dual-port ROM
  - ``"ROM_NP"``: N-port ROM

- **latency** (int, optional): Memory access latency in cycles
- **depth** (int, optional): Depth of the memory (useful for streams/FIFOs)

XLS currently only supports RAM_1RW and RAM_1R1W, so we can only support RAM_1P, RAM_2P, and ROM_1P. The
other storage types are currently not supported.

Conclusion
----------
This example illustrates the process of defining a GEMM kernel using the Allo ADL and generating XLS code 
for the ASIC flow. The approach supports the software synthesis modes which is wonderful for quick verifications and debugging.
