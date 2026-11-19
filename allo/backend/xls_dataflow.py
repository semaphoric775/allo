# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module
"""Dataflow regions on the XLS backend.

Every kernel called by a ``@df.region`` is compiled into its own XLS module
(``XLSCCModule``), so kernels run concurrently. A generated Verilog top wires
region arguments to the kernels and connects each ``allo.stream`` through an
``allo_fifo`` of the declared depth.

Region array arguments are streamed in (``v<k>_in``) when a kernel reads them
and streamed out (``v<k>_out``) when a kernel writes them, in row-major order.
"""

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .._mlir.dialects import allo as allo_d
from .._mlir.ir import Context, Module, StringAttr
from . import efinix, xls_rtl
from .xls import XLSCCModule, _fixed_info

STREAM_RE = re.compile(r"!allo\.stream<(.+),\s*(\d+)>")
FIFO_REG_BITS = 1024  # default: larger stream FIFOs are mapped to block RAM


STORE_RE = re.compile(r"(?:affine|memref)\.store [^\n]*?, %arg(\d+)\[")


def _kernel_module(module_text, kernel, masked=()):
    """Copy of the module keeping `kernel` and the non-kernel helpers.

    Also returns the indices of the kernel arguments it stores to. Arguments in
    `masked` are output with a per-element "written" bit (several writers).
    """
    with Context() as ctx:
        allo_d.register_dialect(ctx)
        mod = Module.parse(module_text, ctx)
        funcs = {
            str(op.attributes["sym_name"]).strip('"'): op
            for op in mod.body.operations
            if op.operation.name == "func.func" and "sym_name" in op.attributes
        }
        # analyse before erasing anything (erasure invalidates printing)
        written = {int(i) for i in STORE_RE.findall(str(funcs[kernel]))}
        if masked:
            funcs[kernel].attributes["xls_masked_outputs"] = StringAttr.get(
                ",".join(str(i) for i in sorted(masked))
            )
        for name, op in funcs.items():
            attrs = op.attributes
            if name != kernel and ("dataflow" in attrs or "df.kernel" in attrs):
                op.operation.erase()
        return str(mod), written


def _parse_region(module_text, top):
    """Streams, kernel calls and argument shapes of the dataflow region `top`."""
    with Context() as ctx:
        allo_d.register_dialect(ctx)
        mod = Module.parse(module_text, ctx)
        func = next(
            op
            for op in mod.body.operations
            if "sym_name" in op.attributes
            and str(op.attributes["sym_name"]).strip('"') == top
        )
        block = func.regions[0].blocks[0]
        args = list(block.arguments)
        arg_types = [str(a.type) for a in args]
        hints = (
            str(func.attributes["itypes"]).strip('"')
            if "itypes" in func.attributes
            else ""
        )
        streams, calls = [], []
        for op in block.operations:
            name = op.operation.name
            if name == "allo.stream_construct":
                m = STREAM_RE.search(str(op.results[0].type))
                sname = str(op.attributes["name"]).strip('"')
                streams.append((op.results[0], sname, int(m.group(2))))
            elif name == "func.call":
                callee = str(op.attributes["callee"]).lstrip("@")
                operands = []
                for v in op.operands:
                    if v in args:
                        operands.append(("arg", args.index(v)))
                    else:
                        idx = [s[0] for s in streams].index(v)
                        operands.append(("stream", idx))
                calls.append((callee, operands))
        streams = [(n, d) for _, n, d in streams]
        signed = [hints[i] != "u" if i < len(hints) else True for i in range(len(args))]
        return arg_types, signed, streams, calls


def _shape_size(type_str):
    m = re.search(r"<((?:\d+x)+)", type_str)
    if not m:
        return 1
    return int(np.prod([int(d) for d in m.group(1).rstrip("x").split("x")]))


class XLSDataflowModule:  # pylint: disable=too-many-instance-attributes
    """RTL (and optionally Verilator cosim) for a dataflow region."""

    def __init__(self, module, top_func_name, project=None, mode="rtl", configs=None):
        configs = dict(configs or {})
        use_memory = configs.pop("use_memory", False)
        # stream FIFOs up to this many bits stay in registers, larger ones
        # go to block RAM (trades LUT/FF usage against RAM blocks)
        self.fifo_reg_bits = configs.pop("fifo_reg_bits", FIFO_REG_BITS)
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project or tempfile.mkdtemp(prefix="xls_df_")
        os.makedirs(self.project, exist_ok=True)
        text = str(module)
        self.arg_types, self.signed, self.streams, self.calls = _parse_region(
            text, top_func_name
        )

        # 1. region arguments written by several kernels need masked outputs
        written = {c: _kernel_module(text, c)[1] for c, _ in self.calls}
        writers = {}
        for callee, operands in self.calls:
            for formal, (kind, idx) in enumerate(operands):
                if kind == "arg" and formal in written[callee]:
                    writers.setdefault(idx, []).append((callee, formal))
        masked = {}
        for ws in writers.values():
            if len(ws) > 1:
                for callee, formal in ws:
                    masked.setdefault(callee, set()).add(formal)

        # 2. one XLS module per kernel
        self.kernels = {}
        sources = []
        for callee, _ in self.calls:
            if callee in self.kernels:
                continue
            kmod = XLSCCModule(
                _kernel_module(text, callee, masked.get(callee, ()))[0],
                callee,
                project=os.path.join(self.project, callee),
                use_memory=use_memory,
                mode="rtl",
                configs=configs,
            )
            self.kernels[callee] = kmod
            sources += kmod.rtl_sources[:2]  # core + wrapper (RAM models shared)
        self.reset = next(iter(self.kernels.values())).configs["reset"]
        ram_v = shutil.copy(xls_rtl.RTL_HARNESS / "allo_ram.v", self.project)

        # 3. top-level netlist
        self.verilog = self._gen_top()
        top_v = os.path.join(self.project, f"{top_func_name}.v")
        with open(top_v, "w", encoding="utf-8") as f:
            f.write(self.verilog)
        self.rtl_sources = sources + [top_v, ram_v]
        self.top_ports = xls_rtl.parse_ports(self.verilog, top_func_name)
        # AXI4-Stream view of the region (tlast after each output array)
        self.axis_source = os.path.join(self.project, f"{top_func_name}_axis.v")
        with open(self.axis_source, "w", encoding="utf-8") as f:
            f.write(
                xls_rtl.gen_axis_wrapper(
                    top_func_name,
                    self.top_ports,
                    {f"v{k}_out": _shape_size(self.arg_types[k]) for k in self.outputs},
                    self.reset,
                )
            )
        if mode == "cosim":
            self._build_cosim()

    def _kernel_chan(self, kmod, formal_idx, kind):
        """Port name of a kernel's formal argument (array index -> v<a>)."""
        if kind == "stream":
            return f"s{formal_idx}"
        array_args = [i for i, a in enumerate(kmod._arg_info) if a["is_array"]]
        return f"v{array_args.index(formal_idx)}"

    def _gen_top(self):
        rst = self.reset
        wires, insts, ports, readers, writers = [], [], {}, {}, {}
        stream_ends = {}  # stream idx -> {"in": (inst, port), "out": ...}
        for inst_idx, (callee, operands) in enumerate(self.calls):
            kmod = self.kernels[callee]
            kports = {n: (d, w) for d, n, w in kmod.top_ports}
            inst = f"u{inst_idx}_{callee}"
            conns = [".clk(clk)"] + ([f".{rst}({rst})"] if rst in kports else [])
            for formal, (kind, idx) in enumerate(operands):
                base = self._kernel_chan(kmod, formal, kind)
                if kind == "stream":
                    d, w = kports[base]
                    end = "in" if d == "input" else "out"
                    if end in stream_ends.setdefault(idx, {}):
                        raise RuntimeError(
                            f"stream {self.streams[idx][0]} has two {end} ends"
                        )
                    net = f"{self.streams[idx][0]}_{'rd' if end == 'in' else 'wr'}"
                    stream_ends[idx][end] = (net, w)
                    conns += [
                        f".{base}({net})",
                        f".{base}_vld({net}_vld)",
                        f".{base}_rdy({net}_rdy)",
                    ]
                    continue
                for suffix in ("_in", "_out"):
                    if base + suffix not in kports:
                        continue
                    net, w = f"{inst}_{base}{suffix}", kports[base + suffix][1]
                    wires.append(f"  wire {xls_rtl._vec(w)}{net};")
                    wires.append(f"  wire {net}_vld, {net}_rdy;")
                    ends = readers if suffix == "_in" else writers
                    ends.setdefault(idx, []).append((net, w))
                    conns += [
                        f".{base}{suffix}({net})",
                        f".{base}{suffix}_vld({net}_vld)",
                        f".{base}{suffix}_rdy({net}_rdy)",
                    ]
            insts.append(
                f"  {callee} {inst} (\n    " + ",\n    ".join(conns) + "\n  );"
            )
        for idx, (name, depth) in enumerate(self.streams):
            ends = stream_ends.get(idx, {})
            if not ends:
                continue  # declared but unused
            if set(ends) != {"in", "out"}:
                raise RuntimeError(
                    f"stream {name} needs exactly one producer and one consumer"
                )
            (rd, w), (wr, _) = ends["in"], ends["out"]
            depth = max(2, 1 << (depth - 1).bit_length())
            aw = depth.bit_length() - 1
            vec = xls_rtl._vec(w)
            wires += [f"  wire {vec}{n};\n  wire {n}_vld, {n}_rdy;" for n in (rd, wr)]
            # shallow FIFOs in registers, deep ones in block RAM
            fifo = "allo_fifo" if depth * w <= self.fifo_reg_bits else "allo_fifo_ram"
            insts.append(
                f"  {fifo} #(.WIDTH({w}), .DEPTH({depth}), .AW({aw})) u_{name} (\n"
                f"    .clk(clk), .rst({rst}),\n"
                f"    .in({wr}), .in_vld({wr}_vld), .in_rdy({wr}_rdy),\n"
                f"    .out({rd}), .out_vld({rd}_vld), .out_rdy({rd}_rdy)\n  );"
            )
        for idx, nets in readers.items():
            insts += self._glue_inputs(f"v{idx}_in", nets, ports)
        for idx, nets in writers.items():
            insts += self._glue_outputs(f"v{idx}_out", nets, ports)
        decls = ["  input wire clk", f"  input wire {rst}"]
        for port, (d, w) in sorted(ports.items()):
            back = "output" if d == "input" else "input"
            decls += [
                f"  {d} wire {xls_rtl._vec(w)}{port}",
                f"  {d} wire {port}_vld",
                f"  {back} wire {port}_rdy",
            ]
        self.inputs = sorted(
            int(p[1:-3]) for p, (d, _) in ports.items() if d == "input"
        )
        self.outputs = sorted(
            int(p[1:-4]) for p, (d, _) in ports.items() if d == "output"
        )
        return (
            f"// Auto-generated by Allo: dataflow region {self.top_func_name}\n"
            f"module {self.top_func_name} (\n"
            + ",\n".join(decls)
            + "\n);\n"
            + "\n".join(wires)
            + "\n"
            + "\n".join(insts)
            + "\nendmodule\n"
        )

    def _glue_inputs(self, port, nets, ports):
        """Drive the kernel inputs `nets` from top-level input `port`."""
        ports[port] = ("input", nets[0][1])
        if len(nets) == 1:
            net = nets[0][0]
            return [
                f"  assign {net} = {port};",
                f"  assign {net}_vld = {port}_vld;",
                f"  assign {port}_rdy = {net}_rdy;",
            ]
        n, w = len(nets), nets[0][1]
        bus = f"{port}_bc"
        lines = [
            f"  wire {xls_rtl._vec(w)}{bus};",
            f"  wire [{n - 1}:0] {bus}_vld, {bus}_rdy;",
            f"  allo_bcast #(.WIDTH({w}), .N({n})) u_{bus} (\n"
            f"    .clk(clk), .rst({self.reset}), .in({port}), .in_vld({port}_vld),"
            f" .in_rdy({port}_rdy),\n"
            f"    .out({bus}), .out_vld({bus}_vld), .out_rdy({bus}_rdy)\n  );",
        ]
        for i, (net, _) in enumerate(nets):
            lines += [
                f"  assign {net} = {bus};",
                f"  assign {net}_vld = {bus}_vld[{i}];",
                f"  assign {bus}_rdy[{i}] = {net}_rdy;",
            ]
        return lines

    def _glue_outputs(self, port, nets, ports):
        """Drive top-level output `port` from the kernel outputs `nets`."""
        if len(nets) == 1:
            net, w = nets[0]
            ports[port] = ("output", w)
            return [
                f"  assign {port} = {net};",
                f"  assign {port}_vld = {net}_vld;",
                f"  assign {net}_rdy = {port}_rdy;",
            ]
        n, w = len(nets), nets[0][1] - 1  # writers carry {written, data}
        ports[port] = ("output", w)
        bus = f"{port}_mg"
        data = ", ".join(net for net, _ in reversed(nets))
        lines = [
            f"  wire [{n - 1}:0] {bus}_vld, {bus}_rdy;",
            f"  allo_merge #(.WIDTH({w}), .N({n})) u_{bus} (\n"
            f"    .in({{{data}}}), .in_vld({bus}_vld), .in_rdy({bus}_rdy),\n"
            f"    .out({port}), .out_vld({port}_vld), .out_rdy({port}_rdy)\n  );",
        ]
        for i, (net, _) in enumerate(nets):
            lines += [
                f"  assign {bus}_vld[{i}] = {net}_vld;",
                f"  assign {net}_rdy = {bus}_rdy[{i}];",
            ]
        return lines

    def _build_cosim(self):
        tb = os.path.join(self.project, "tb.cpp")
        cfg = next(iter(self.kernels.values())).configs
        with open(tb, "w", encoding="utf-8") as f:
            f.write(
                xls_rtl.gen_tb(
                    self.top_func_name,
                    self.top_ports,
                    [f"v{k}_in" for k in self.inputs],
                    [f"v{k}_out" for k in self.outputs],
                    n_in=[_shape_size(self.arg_types[k]) for k in self.inputs],
                    n_out=[_shape_size(self.arg_types[k]) for k in self.outputs],
                    stream=True,
                    reset=self.reset,
                    max_cycles=cfg["max_cycles"],
                )
            )
        self._binary = xls_rtl.build_cosim(
            self.project, self.top_func_name, self.rtl_sources, tb
        )

    def __call__(self, *args):
        """Run the RTL; output arguments are written in place."""
        if self.mode != "cosim":
            raise RuntimeError("__call__ is only supported in cosim mode")
        for k, arg_idx in enumerate(self.inputs):
            arg = args[arg_idx]
            fixed = _fixed_info(self.arg_types[arg_idx])
            if fixed:  # real value -> raw fixed-point bits
                arg = np.fix(np.asarray(arg, np.float64) * 2 ** fixed[2])
            np.asarray(arg).astype(np.int64).flatten().tofile(
                os.path.join(self.project, f"input{k}.data")
            )
        res = subprocess.run(
            [self._binary],
            cwd=self.project,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"cosim failed:\n{res.stderr}{res.stdout}")
        m = re.search(r"CYCLES:(\d+)", res.stdout)
        self.cycles = int(m.group(1)) if m else None
        widths = {n: w for _, n, w in self.top_ports}
        for k, arg_idx in enumerate(self.outputs):
            name = "output.data" if len(self.outputs) == 1 else f"output{k}.data"
            out = np.fromfile(os.path.join(self.project, name), np.int64)
            w = widths[f"v{arg_idx}_out"]
            fixed = _fixed_info(self.arg_types[arg_idx])
            if (fixed[0] if fixed else self.signed[arg_idx]) and w < 64:
                out = np.where(out >= (1 << (w - 1)), out - (1 << w), out)
            if fixed:
                out = out.astype(np.float64) / 2 ** fixed[2]
            np.copyto(
                args[arg_idx], out.reshape(np.shape(args[arg_idx])), casting="unsafe"
            )

    def export_efinity(
        self, family="Titanium", device="Ti60F225", flow=None, interface="native", **kw
    ):
        """Efinity project for the region (see XLSCCModule.export_efinity)."""
        top, sources = self.top_func_name, list(self.rtl_sources)
        if interface == "axis":
            top, sources = f"{top}_axis", sources + [self.axis_source]
            kw.setdefault("clock_port", "aclk")
        elif interface != "native":
            raise ValueError(f"Unknown interface {interface!r}")
        out = os.path.join(self.project, "efinity")
        xml = efinix.gen_project(out, top, sources, family=family, device=device, **kw)
        return efinix.run_flow(out, top, flow) if flow else xml
