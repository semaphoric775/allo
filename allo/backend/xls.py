# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module,no-member

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .._mlir.dialects import allo as allo_d
from .._mlir.ir import Context, InsertionPoint, Location, Module, UnitAttr
from .._mlir.passmanager import PassManager
from ..ir.transform import find_func_in_module
from ..passes import decompose_library_function, _mlir_lower_pipeline
from . import efinix, xls_rtl

# Regex patterns for floats, dynamic shapes, and fixed-point types
FLOAT_RE = re.compile(r"\b(f16|f32|f64|bf16)\b")
DYNAMIC_RE = re.compile(r"\b(memref|tensor)<[^>]*\?>")
FIXED_TYPE_RE = re.compile(r"!allo\.(Fixed|UFixed)<\d+,\s*\d+>")

# Manually designed Fixed Point Struct modeled via Fixed Integer type
# Emitted with rest of code when fixed-point types are used in the MLIR
FIXED_POINT_STRUCT = """
// XLS-compatible Fixed-Point Template
template <int WIDTH, int FRAC, bool SIGNED = true>
struct Fixed {
  ac_int<WIDTH, SIGNED> value;
  Fixed() : value(0) {}
  Fixed(int v) : value(static_cast<ac_int<WIDTH, SIGNED>>(v) << FRAC) {}
  static Fixed from_raw(ac_int<WIDTH, SIGNED> raw) { Fixed f; f.value = raw; return f; }
  int to_int() const { return static_cast<int>(value >> FRAC); }
  ac_int<WIDTH, SIGNED> raw() const { return value; }
  Fixed operator+(const Fixed& o) const { Fixed r; r.value = value + o.value; return r; }
  Fixed operator-(const Fixed& o) const { Fixed r; r.value = value - o.value; return r; }
  Fixed operator*(const Fixed& o) const {
    Fixed r;
    ac_int<WIDTH*2, SIGNED> t = static_cast<ac_int<WIDTH*2, SIGNED>>(value) *
                                 static_cast<ac_int<WIDTH*2, SIGNED>>(o.value);
    r.value = static_cast<ac_int<WIDTH, SIGNED>>(t >> FRAC);
    return r;
  }
  Fixed operator/(const Fixed& o) const {
    Fixed r;
    ac_int<WIDTH*2, SIGNED> t = static_cast<ac_int<WIDTH*2, SIGNED>>(value) << FRAC;
    r.value = static_cast<ac_int<WIDTH, SIGNED>>(t / o.value);
    return r;
  }
  bool operator==(const Fixed& o) const { return value == o.value; }
  bool operator<(const Fixed& o) const { return value < o.value; }
};
template <int WIDTH, int FRAC> using UFixed = Fixed<WIDTH, FRAC, false>;
"""


#  Check if g++ (for sw_emu) or XLS toolchain is available
def is_available():
    """Check if g++ (for sw_emu) or XLS toolchain is available."""
    return shutil.which("g++") is not None or shutil.which("xlscc") is not None


def _validate_xls_ir(mlir_text, project=None):
    errors = []
    for i, line in enumerate(mlir_text.splitlines(), 1):
        # Float types
        m = FLOAT_RE.search(line)
        if m:
            errors.append(
                f"Line {i}: Floating-point type '{m.group()}' is not supported by XLS [CC] backend. "
                "Please use integer or fixed-point types instead."
            )
        # Dynamic shapes (memref with ? dimension)
        if DYNAMIC_RE.search(line):
            errors.append(
                f"Line {i}: Dynamic shapes (with '?' dimension) are not supported by the XLS [CC] backend. "
                "All array dimensions must be statically known."
            )

    if errors:
        err_msg = f"XLS [CC] validation failed ({len(errors)} errors):\n" + "\n".join(
            errors
        )
        # write error message to file if project is provided
        if project:
            os.makedirs(project, exist_ok=True)
            with open(f"{project}/xls_errors.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(errors))
            err_msg += f"\nSee {project}/xls_errors.txt"
        raise RuntimeError(err_msg)  # raise error


def _extract_body(code, func_name):
    func_head = rf"void\s+{re.escape(func_name)}\s*\(\s*\)\s*\{{"
    match = re.search(func_head, code)
    if not match:
        return f"// Could not extract body of {func_name}\n"

    # use brace matching to extract the function body
    start, depth, pos = match.end(), 1, match.end()
    while pos < len(code) and depth > 0:
        if code[pos] == "{":
            depth += 1
        elif code[pos] == "}":
            depth -= 1
        pos += 1

    return (
        code[start : pos - 1] if depth == 0 else f"// Unmatched braces in {func_name}\n"
    )


def _reindent(body, indent="    "):
    lines = body.split("\n")
    # Very large number, very unlikely to have that many indentations in the body
    min_indent = 2**20

    for ln in lines:
        if ln.strip():
            min_indent = min(len(ln) - len(ln.lstrip()), min_indent)

    # Set to 0 if no indentation is found
    if min_indent == 2**20:
        min_indent = 0

    result = []
    for ln in lines:
        if not ln.strip():
            result.append("")
        else:
            result.append(indent + ln[min_indent:])

    return "\n".join(result)


def _parse_memory_comments(body):
    # Regex patterns __xls_memory_decl__ and __xls_state_vars__
    # The element type may itself contain commas (e.g. "ac_int<12, true>")
    mem_re = re.compile(
        r"^\s*//\s*__xls_memory_decl__:\s*(.*?),\s*(\d+),\s*(\w+),\s*(\d+)"
        r"((?:,\s*\d+)*)"
    )
    # Completely partitioned arrays: class-level registers, "type, name, dims"
    arr_re = re.compile(
        r"^\s*//\s*__xls_array_decl__:\s*(.*),\s*([A-Za-z_]\w*)((?:,\s*\d+)+)\s*;?\s*$"
    )
    state_re = re.compile(r"^\s*//\s*__xls_state_vars__:\s*(.+)$")

    # Collect memory declarations, state variables.
    decls, states = [], []
    mems = []  # Information about memories to generate XLS's RAM Config Files
    result = []  # final result with cleaned up body.

    for line in body.split("\n"):
        mem_m = mem_re.match(line)
        arr_m = arr_re.match(line)
        state_m = state_re.match(line)

        if arr_m:
            elem_type, name = arr_m.group(1).strip(), arr_m.group(2)
            shape = "".join(f"[{d}]" for d in re.findall(r"\d+", arr_m.group(3)))
            decls.append(f"{elem_type} {name}{shape};")
        elif mem_m:
            # Format: elem_type, size, name, memory_space, dim0, dim1, ...
            elem_type, size_str, name, mem_space_str = (
                mem_m.group(1).strip(),
                mem_m.group(2),
                mem_m.group(3),
                mem_m.group(4),
            )
            dims = re.findall(r"\d+", mem_m.group(5))

            decls.append(f"__xls_memory<{elem_type}, {size_str}> {name};")

            size_m = re.match(r"(\d+)", size_str)
            mem_space_m = re.match(r"(\d+)", mem_space_str)
            if size_m:
                size = int(size_m.group(1))
                memory_space = int(mem_space_m.group(1)) if mem_space_m else 0
                # Decode: memory_space = resource_code * 16 + storage_type_code
                storage_type_code = memory_space % 16
                # Default to RAM_2P (dual port, maps to RAM_1R1W in XLS) when not specified
                if storage_type_code == 0:
                    storage_type_code = 2

                int_dims = [
                    int(re.match(r"(\d+)", d).group(1))
                    for d in dims
                    if re.match(r"(\d+)", d)
                ]
                mems.append(
                    (name, elem_type, size, int_dims or [size], storage_type_code)
                )

        elif state_m:
            for v in state_m.group(1).split(","):
                v = v.split(";")[0].split("//")[0].strip()
                if v:
                    states.append(f"int {v} = 0;")
        else:
            result.append(line)

    return decls, states, "\n".join(result), mems


# Map storage_type_code to XLS RAM kind. Default is RAM_2P.
STORAGE_TYPE_TO_XLS_RAM = {
    1: "RAM_1RW",  # RAM_1P (single port)
    2: "RAM_1R1W",  # RAM_2P (dual port) - default
    5: "RAM_1R1W",  # RAM_S2P (alias of RAM_2P)
    6: "RAM_1RW",  # ROM_1P (single port ROM)
}

# Mapping from storage_type_code to human-readable name for error messages
STORAGE_TYPE_CODE_TO_NAME = {
    1: "RAM_1P",
    2: "RAM_2P",
    3: "RAM_T2P",
    4: "RAM_1WNR",
    5: "RAM_S2P",
    6: "ROM_1P",
    7: "ROM_2P",
    8: "ROM_NP",
}

# Supported storage types for XLS backend (1=RAM_1P, 2=RAM_2P, 6=ROM_1P)
SUPPORTED_STORAGE_TYPES = {1, 2, 6}


# Generates the RAM configuration required by XLS for the memory interface
def _gen_textproto(mems):
    if not mems:
        return ""
    lines = [
        "# Automatically Generated by Allo (Allo XLS [CC] Backend)",
        "# proto-file: xls/codegen/ram_rewrite.proto",
        "# proto-message: Ram Configuration Files",
        "",
    ]
    for name, _, size, _, storage_type_code in mems:
        # Validate storage_type_code is supported by XLS backend
        if storage_type_code not in SUPPORTED_STORAGE_TYPES:
            storage_type_name = STORAGE_TYPE_CODE_TO_NAME.get(
                storage_type_code, f"Unknown (code {storage_type_code})"
            )
            supported_names = ", ".join(
                STORAGE_TYPE_CODE_TO_NAME.get(c, str(c))
                for c in sorted(SUPPORTED_STORAGE_TYPES)
            )
            raise RuntimeError(
                f"XLS [CC] validation failed: Memory '{name}' uses storage_type '{storage_type_name}' "
                f"which is not supported by the XLS backend. "
                f"XLS only supports RAM_1RW and RAM_1R1W configurations. "
                f"Supported storage_types are: {supported_names}, or leave unspecified for default (RAM_1R1W)."
            )

        # Determine XLS RAM kind based on storage_type_code
        xls_ram_kind = STORAGE_TYPE_TO_XLS_RAM[storage_type_code]

        # Use size (array size) as depth by default
        depth = size

        # Determine channel mappings based on RAM kind
        if xls_ram_kind == "RAM_1RW":
            # Single port RAM - shared channels for read and write
            read_req = f"{name}_req"
            read_resp = f"{name}_resp"
            write_req = f"{name}_req"
            write_resp = f"{name}_resp"
        else:
            # Dual port RAM (RAM_1R1W) - separate read and write channels
            read_req = f"{name}_read_request"
            read_resp = f"{name}_read_response"
            write_req = f"{name}_write_request"
            write_resp = f"{name}_write_response"

        # Generate rewrite configuration (only difference is to_config kind)
        lines.append(
            f"""rewrites {{
  from_config {{ kind: RAM_ABSTRACT depth: {depth} }}
  to_config {{ kind: {xls_ram_kind} depth: {depth} }}
  from_channels_logical_to_physical: {{ key: "abstract_read_req" value: "{read_req}" }}
  from_channels_logical_to_physical: {{ key: "abstract_read_resp" value: "{read_resp}" }}
  from_channels_logical_to_physical: {{ key: "abstract_write_req" value: "{write_req}" }}
  from_channels_logical_to_physical: {{ key: "write_completion" value: "{write_resp}" }}
  to_name_prefix: "{name}_"
}}
"""
        )
    return "\n".join(lines)


# Render TestBlock class with channels and optional memory/state declarations.
def _render_testblock(
    in_chans,
    out_chans,
    body,
    top_name,
    mem_decls=None,
    state_decls=None,
    *,
    chan_types=None,
):
    # Templated channels for inputs and outputs (element type, default int)
    types = chan_types or {}

    def decl(chan, direction):
        return f"__xls_channel<{types.get(chan, 'int')}, __xls_channel_dir_{direction}> {chan};"

    in_decl = "\n  ".join(decl(c, "In") for c in in_chans)
    out_decl = "\n  ".join(decl(c, "Out") for c in out_chans)
    indented = _reindent(body)  # reindent body to match TestBlock class indentation

    mem_section = ""
    if mem_decls:
        mem_section = "\n  // Memory\n  " + "\n  ".join(mem_decls) + "\n"

    state_section = ""
    if state_decls:
        state_section = "\n  // State\n  " + "\n  ".join(state_decls) + "\n"

    return f"""class TestBlock {{
public:
  {in_decl}
  {out_decl}
{mem_section}{state_section}
  #pragma hls_top
  void {top_name}() {{
{indented}
  }}
}};
"""


FIXED_INFO_RE = re.compile(r"!allo\.(U?)Fixed<(\d+),\s*(\d+)>")
FIXED_ANY_RE = re.compile(r"!allo\.U?Fixed<(\d+),\s*\d+>")


def _fixed_info(type_str):
    """(signed, width, frac) of a fixed-point (element) type, else ()."""
    m = FIXED_INFO_RE.search(type_str)
    return (not m.group(1), int(m.group(2)), int(m.group(3))) if m else ()


def _cast_stream_puts(module):
    """Convert values put into fixed-point streams to the stream's element type.

    The frontend may put e.g. a Fixed<32, 16> product into a Fixed<16, 8>
    stream, relying on C++ assignment; once fixed-point is lowered to raw
    integers the conversion has to be explicit.
    """
    puts = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.operation.name == "allo.stream_put":
                        puts.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    for put in puts:
        value = put.operands[-1]
        elem = allo_d.StreamType(put.operands[0].type).base_type
        if value.type == elem or not FIXED_INFO_RE.search(str(elem)):
            continue
        cast_op = (
            allo_d.FixedToFixedOp
            if FIXED_INFO_RE.search(str(value.type))
            else allo_d.IntToFixedOp
        )
        with InsertionPoint(put):
            cast = cast_op(elem, value)
        put.operands[len(put.operands) - 1] = cast.result


def _channel_ctype(type_str, hint):
    """C++ type carried by a channel for an MLIR memref/tensor element type."""
    m = re.search(r"<(?:\d+x)*u?i(\d+)\b", type_str)
    if not m:
        return "int"  # index / fixed-point: keep the integer interface
    width, signed = int(m.group(1)), hint != "u"
    if width == 32 and signed:
        return "int"
    return f"ac_int<{width}, {'true' if signed else 'false'}>"


# Channels announced by the emitter for dataflow kernels: "type, name, In|Out"
CHAN_DECL_RE = re.compile(
    r"^\s*//\s*__xls_chan_decl__:\s*(.*),\s*([A-Za-z_]\w*),\s*(In|Out)\s*$", re.M
)


def _wrap_xlscc(core_code, top_name, func_name, inputs, use_memory, *, chan_types=None):
    body = _extract_body(core_code, func_name)
    chans = CHAN_DECL_RE.findall(body)
    # No array inputs -> combinational, return without channels, memory, TestBlock class
    if not inputs and not chans:
        return core_code, "", []

    # Extract headers
    match = re.search(rf"void\s+{re.escape(func_name)}\s*\(\s*\)\s*\{{", core_code)
    headers = core_code[: match.start()] if match else ""

    # Parse memory declarations from body
    mem_decls, state_decls, result, mems = _parse_memory_comments(body)

    # Generate channel names
    in_chans = [f"v{i}_in" for i in range(len(inputs))]
    out_chans = ["out"]
    if chans:  # dataflow kernel: exactly the channels the emitter declared
        in_chans = [n for _, n, d in chans if d == "In"]
        out_chans = [n for _, n, d in chans if d == "Out"]
        chan_types = {n: t.strip() for t, n, _ in chans}

    # Render testblock
    if use_memory:
        testblock = _render_testblock(
            in_chans,
            out_chans,
            result,
            top_name,
            mem_decls,
            state_decls,
            chan_types=chan_types,
        )
        textproto = _gen_textproto(mems)
    else:
        testblock = _render_testblock(
            in_chans, out_chans, body, top_name, chan_types=chan_types
        )
        textproto = ""
        mems = []

    return headers + testblock, textproto, mems


class XLSCCModule:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        mlir_text_or_module,
        top_func_name,
        project=None,
        use_memory=False,
        mode=None,
        *,
        configs=None,
    ):
        self.top_func_name = top_func_name
        self.configs = {**xls_rtl.DEFAULT_RTL_CONFIGS, **(configs or {})}
        self.project = project
        self.use_memory = use_memory
        self.mode = mode
        self._binary_path = None
        self._project_dir = project
        self._arg_info = []
        self._has_arrays = False
        self._output_info = None
        self.rtl_sources = []

        # Parse MLIR + run minimal lowering
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mlir_text_or_module), ctx)

            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                "empty-tensor-to-alloc-tensor,"
                "func.func(convert-linalg-to-affine-loops)"
                ")"
            )
            pm.run(self.module.operation)

            self.func = find_func_in_module(self.module, top_func_name)
            # Fixed-point interface types, recorded before they are lowered
            self._fixed_args, self._fixed_out, self._out_frac = [], None, None
            if self.func:
                self._fixed_args = [
                    _fixed_info(str(a.type)) for a in self.func.arguments
                ]
                results = list(self.func.type.results)
                fixed_out = _fixed_info(str(results[0])) if results else ()
                self._fixed_out = fixed_out
                # fraction bits of a fixed-point result (None otherwise)
                self._out_frac = fixed_out[2] if fixed_out else None
            if mode in {"rtl", "cosim"}:
                # Hardware carries fixed-point values as raw integers (the same
                # lowering the LLVM backend uses, so results match bit for bit)
                _cast_stream_puts(self.module)
                allo_d.lower_fixed_to_int(self.module)
                # the pass leaves stream element types (and stream_get/put
                # signatures) alone; their values are raw integers by now
                text = FIXED_ANY_RE.sub(r"i\1", str(self.module))
                self.module = Module.parse(text, ctx)
                self.func = find_func_in_module(self.module, top_func_name)
            if self.func:
                self.func.attributes["top"] = UnitAttr.get()

        self.mlir_text = str(self.module)

        # validate XLS IR does not contain unsupported features
        _validate_xls_ir(self.mlir_text, project)

        # Emit Core XLS C++ from MLIR
        buf = io.StringIO()
        allo_d.emit_xhls(self.module, buf, use_memory)
        buf.seek(0)
        self.core_code = buf.read()

        # Extract argument info for sw_emu
        self._extract_arg_info()

        # Get array parameter names and the element type each channel carries
        param_names = []
        chan_types = {}
        if self.func:
            itypes = self._type_hints("itypes")
            for i, arg in enumerate(self.func.arguments):
                t = str(arg.type)
                if ("memref" in t or "tensor" in t) and "stream" not in t.lower():
                    chan_types[f"v{len(param_names)}_in"] = _channel_ctype(
                        t, itypes[i] if i < len(itypes) else "s"
                    )
                    param_names.append(f"arg{len(param_names)}")
            results = list(self.func.type.results)
            if results:
                chan_types["out"] = _channel_ctype(
                    str(results[0]), (self._type_hints("otypes") or "s")[0]
                )

        # Add fixed-point struct if needed
        fixed_header = (
            FIXED_POINT_STRUCT if FIXED_TYPE_RE.search(self.mlir_text) else ""
        )

        # Use _wrap_xlscc to handle C++ templated memory/channels, headers, top pragma, etc. as well
        # as RAM Configuration File (rewrites.textproto) which are not handled by emit_xls
        cpp, self.rewrites_textproto, self.mems = _wrap_xlscc(
            self.core_code,
            top_func_name,
            top_func_name,
            param_names,
            use_memory,
            chan_types=chan_types,
        )
        self.final_cpp = fixed_header + cpp

        # Write output (C++ and RAM Configuration File) files if project is provided
        if project:
            os.makedirs(project, exist_ok=True)
            with open(f"{project}/test_block.cpp", "w", encoding="utf-8") as f:
                f.write(self.final_cpp)
            if use_memory and self.rewrites_textproto:
                with open(f"{project}/rewrites.textproto", "w", encoding="utf-8") as f:
                    f.write(self.rewrites_textproto)

        # Build sw_emu binary if requested
        if mode == "sw_emu":
            self._build_sw_emu()
        elif mode in {"rtl", "cosim"}:
            self._build_rtl()
            if mode == "cosim":
                self._build_cosim()

    def _build_rtl(self):
        """xlscc -> opt_main -> codegen_main, then wrap with RAM models."""
        project = self.project or tempfile.mkdtemp(prefix="xls_rtl_")
        self._project_dir = project
        os.makedirs(project, exist_ok=True)
        cpp_path = os.path.join(project, "test_block.cpp")
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(self.final_cpp)
        rewrites = None
        if self.rewrites_textproto:
            rewrites = os.path.join(project, "rewrites.textproto")
            with open(rewrites, "w", encoding="utf-8") as f:
                f.write(self.rewrites_textproto)
        self._is_stream = "class TestBlock" in self.final_cpp
        ir = xls_rtl.cpp_to_ir(
            cpp_path,
            project,
            self._is_stream,
            self.configs["include_dirs"],
            self.configs["xlscc_flags"],
        )
        mems = [
            (name, STORAGE_TYPE_TO_XLS_RAM[code], size)
            for name, _, size, _, code in self.mems
        ]
        core = f"{self.top_func_name}_core"
        core_v, self.schedule = xls_rtl.ir_to_verilog(
            ir,
            project,
            core,
            self.configs,
            rewrites=rewrites,
            mems=[m[:2] for m in mems],
        )
        with open(core_v, encoding="utf-8") as f:
            self.core_ports = xls_rtl.parse_ports(f.read(), core)
        top_v = os.path.join(project, f"{self.top_func_name}.v")
        self.verilog = xls_rtl.gen_top(core, self.core_ports, mems, self.top_func_name)
        with open(top_v, "w", encoding="utf-8") as f:
            f.write(self.verilog)
        ram_v = shutil.copy(xls_rtl.RTL_HARNESS / "allo_ram.v", project)
        self.rtl_sources = [core_v, top_v, ram_v]
        self.top_ports = xls_rtl.parse_ports(self.verilog, self.top_func_name)
        self.axis_source = None
        if self._is_stream:
            # AXI4-Stream view for DMA / SoC integration (e.g. Efinix Sapphire)
            self.axis_source = os.path.join(project, f"{self.top_func_name}_axis.v")
            with open(self.axis_source, "w", encoding="utf-8") as f:
                f.write(
                    xls_rtl.gen_axis_wrapper(
                        self.top_func_name,
                        self.top_ports,
                        (self._output_info or {"size": 1})["size"],
                        self.configs["reset"],
                    )
                )

    def _build_cosim(self):
        if shutil.which("verilator") is None:
            raise RuntimeError("verilator not found. Cannot build cosim.")
        ctrl = {"clk", self.configs["reset"], "in_vld"}
        if self._is_stream:
            n_in = [a["size"] for a in self._arg_info if a["is_array"]]
            ins = [f"v{k}_in" for k in range(len(n_in))]
            out, n_out = "out", (self._output_info or {"size": 1})["size"]
        else:
            ins = [n for d, n, _ in self.top_ports if d == "input" and n not in ctrl]
            outs = [n for d, n, _ in self.top_ports if d == "output" and n != "out_vld"]
            n_in, out, n_out = [1] * len(ins), outs[0], 1
        self._out_width = {n: w for _, n, w in self.top_ports}[out]
        self._out_signed = (self._type_hints("otypes") or "s")[0] != "u"
        tb = os.path.join(self._project_dir, "tb.cpp")
        with open(tb, "w", encoding="utf-8") as f:
            f.write(
                xls_rtl.gen_tb(
                    self.top_func_name,
                    self.top_ports,
                    ins,
                    out,
                    n_in=n_in,
                    n_out=n_out,
                    stream=self._is_stream,
                    reset=self.configs["reset"],
                    max_cycles=self.configs["max_cycles"],
                )
            )
        self._binary_path = xls_rtl.build_cosim(
            self._project_dir, self.top_func_name, self.rtl_sources, tb
        )

    def _type_hints(self, attr):
        """Per-argument signedness hints ("s"/"u"/"_") attached by the builder."""
        if self.func and attr in self.func.attributes:
            return str(self.func.attributes[attr]).strip('"')
        return ""

    def _extract_arg_info(self):
        if not self.func:
            return

        for i, arg in enumerate(self.func.arguments):
            t = str(arg.type)
            info = {"index": i, "type_str": t, "is_array": False}

            if "memref" in t or "tensor" in t:
                info["is_array"] = True
                self._has_arrays = True
                match = re.search(r"<((?:\d+x)+)([^,>]+(?:<[^>]*>)?)", t)
                if match:
                    info["shape"] = [
                        int(p) for p in match.group(1).rstrip("x").split("x")
                    ]
                    info["dtype"] = match.group(2)
                    info["size"] = 1
                    for dim in info["shape"]:
                        info["size"] *= dim
            else:
                info["dtype"] = t

            self._arg_info.append(info)

        # Extract output info from function return type
        self._extract_output_info()

    # Extract output shape and dtype from function return type.
    def _extract_output_info(self):
        if not self.func:
            return
        result_types = list(self.func.type.results)
        if not result_types:
            return
        ret_t = str(result_types[0])
        if "memref" not in ret_t and "tensor" not in ret_t:
            return
        match = re.search(r"<([^>]+)>", ret_t)
        if not match:
            return
        parts = match.group(1).split("x")
        if len(parts) < 2:
            return
        self._output_info = {
            "shape": [int(p) for p in parts[:-1]],
            "dtype": parts[-1],
        }
        self._output_info["size"] = 1
        for dim in self._output_info["shape"]:
            self._output_info["size"] *= dim

    def _get_harness_path(self):
        return Path(__file__).parent.parent / "harness" / "xlscc"

    def _preprocess_cpp(self, code):
        lines = []
        all_lines = code.split("\n")
        skip_next = False
        for i, line in enumerate(all_lines):
            if skip_next:
                skip_next = False
                continue
            # Skip XLS-specific includes
            if "#include" in line and (
                "xls" in line.lower() or "ac_int" in line or "ac_channel" in line
            ):
                continue
            # Skip template line if next line is using ac_int (they come as a pair)
            if line.strip().startswith("template") and i + 1 < len(all_lines):
                next_line = all_lines[i + 1]
                if "using ac_int" in next_line or "using ac_uint" in next_line:
                    skip_next = True
                    continue
            # Skip using ac_int/ac_uint lines
            if "using ac_int" in line or "using ac_uint" in line:
                continue
            lines.append(line)
        return "\n".join(lines)

    def _generate_harness_scalar(self):
        harness_path = self._get_harness_path()
        processed = self._preprocess_cpp(self.final_cpp)

        harness = f"""// Auto-generated sw_emu harness (scalar)
#include "{harness_path}/XlsInt.h"
#include "{harness_path}/Channel.h"
#include "{harness_path}/Memory.h"
#include <iostream>
#include <cstdlib>

{processed}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <inputs...>" << std::endl;
        return 1;
    }}

"""
        args = []
        for i, info in enumerate(self._arg_info):
            if not info["is_array"]:
                harness += f"    int arg{i} = std::atoi(argv[{i + 1}]);\n"
                args.append(f"arg{i}")

        if args:
            harness += f"\n    auto result = {self.top_func_name}({', '.join(args)});\n"
            harness += '    std::cout << "RESULT:" << result << std::endl;\n'
        else:
            harness += f"\n    {self.top_func_name}();\n"
            harness += '    std::cout << "RESULT:OK" << std::endl;\n'

        harness += "    return 0;\n}\n"
        return harness

    def _generate_harness_array(self):
        harness_path = self._get_harness_path()
        processed = self._preprocess_cpp(self.final_cpp)
        output_size = self._output_info["size"] if self._output_info else 1

        harness = f"""// Auto-generated sw_emu harness (array)
#include "{harness_path}/XlsInt.h"
#include "{harness_path}/Channel.h"
#include "{harness_path}/Memory.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

{processed}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <project_dir>" << std::endl;
        return 1;
    }}
    std::string project_dir = argv[1];
    TestBlock block;

"""
        for i, info in enumerate(self._arg_info):
            if info["is_array"]:
                size = info.get("size", 1)
                harness += f"""
    std::ifstream in{i}(project_dir + "/input{i}.data", std::ios::binary);
    if (!in{i}.is_open()) {{
        std::cerr << "Failed to open input{i}.data" << std::endl;
        return 1;
    }}
    std::vector<int32_t> input{i}({size});
    in{i}.read(reinterpret_cast<char*>(input{i}.data()), {size} * sizeof(int32_t));
    in{i}.close();
    for (int j = 0; j < {size}; ++j) block.v{i}_in.write(input{i}[j]);
"""

        harness += f"""
    block.{self.top_func_name}();

    std::cout << "RESULT:";
    for (int j = 0; j < {output_size}; ++j) {{
        int32_t val = block.out.read();
        std::cout << val;
        if (j < {output_size} - 1) std::cout << ",";
    }}
    std::cout << std::endl;
    return 0;
}}
"""
        return harness

    def _generate_makefile(self):
        """Generate a Makefile for the sw_emu project."""
        harness_path = self._get_harness_path()
        makefile = f"""# Auto-generated Makefile for XLS sw_emu
# Generated by Allo XLS [CC] Backend

CXX = g++
CXXFLAGS = -std=c++17 -O2
INCLUDES = -I{harness_path}

TARGET = test_binary
SRCS = test_harness.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
\t$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

%.o: %.cpp
\t$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

run: $(TARGET)
"""
        if self._has_arrays:
            makefile += "\t./$(TARGET) .\n"
        else:
            makefile += "\t./$(TARGET)\n"

        makefile += """
clean:
\trm -f $(TARGET) $(OBJS)

help:
\t@echo "Makefile targets:"
\t@echo "  all   - Build the test binary (default)"
\t@echo "  run   - Build and run the test binary"
\t@echo "  clean - Remove build artifacts"
\t@echo "  help  - Show this help message"
"""
        return makefile

    def _build_sw_emu(self):
        if shutil.which("g++") is None:
            raise RuntimeError("g++ not found. Cannot build sw_emu.")

        project_dir = self.project or tempfile.mkdtemp(prefix="xls_sw_emu_")
        self._project_dir = project_dir
        os.makedirs(project_dir, exist_ok=True)

        harness = (
            self._generate_harness_array()
            if self._has_arrays
            else self._generate_harness_scalar()
        )
        harness_path = os.path.join(project_dir, "test_harness.cpp")
        binary_path = os.path.join(project_dir, "test_binary")

        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness)

        # Generate Makefile
        makefile = self._generate_makefile()
        makefile_path = os.path.join(project_dir, "Makefile")
        with open(makefile_path, "w", encoding="utf-8") as f:
            f.write(makefile)

        result = subprocess.run(
            ["g++", "-std=c++17", "-O2", "-o", binary_path, harness_path],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"sw_emu compilation failed:\n{result.stderr}\n{result.stdout}"
            )

        self._binary_path = binary_path

    def __call__(self, *args):
        if self.mode == "cosim":
            return self._call_cosim(*args)
        if self.mode != "sw_emu":
            raise RuntimeError("__call__ only supported in sw_emu and cosim modes")
        if not self._binary_path or not os.path.exists(self._binary_path):
            raise RuntimeError("sw_emu binary not built")

        if self._has_arrays:
            return self._call_array(*args)
        return self._call_scalar(*args)

    def _call_scalar(self, *args):
        str_args = [str(a) for a in args]
        result = subprocess.run(
            [self._binary_path] + str_args, capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"sw_emu execution failed:\n{result.stderr}")

        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                val = line[7:]
                if val == "OK":
                    return None
                try:
                    return int(val)
                except ValueError:
                    return val
        return None

    def _call_array(self, *args):
        for i, (arg, info) in enumerate(zip(args, self._arg_info)):
            if info["is_array"]:
                arr = np.asarray(arg, dtype=np.int32).flatten()
                with open(f"{self._project_dir}/input{i}.data", "wb") as f:
                    f.write(arr.tobytes())

        result = subprocess.run(
            [self._binary_path, self._project_dir],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"sw_emu execution failed:\n{result.stderr}")

        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                val = line[7:]
                if val == "OK":
                    return None
                vals = [int(v) for v in val.split(",")]
                if self._output_info:
                    return np.array(vals, dtype=np.int32).reshape(
                        self._output_info["shape"]
                    )
                return np.array(vals, dtype=np.int32)
        return None

    def export_efinity(
        self, family="Titanium", device="Ti60F225", flow=None, interface="native", **kw
    ):
        """Write an Efinity project for the generated RTL to ``<project>/efinity``.

        ``interface``: "native" (XLS ready/valid ports) or "axis" (AXI4-Stream,
        streaming designs only). ``flow`` ("map" or "pnr") optionally runs
        Efinity and returns its resource report (and ``fmax_mhz`` after pnr);
        otherwise returns the project XML path. ``kw``: ``timing_model``,
        ``clock_mhz``.
        """
        if not self.rtl_sources:
            raise RuntimeError("export_efinity requires mode='rtl' or 'cosim'")
        top, sources = self.top_func_name, list(self.rtl_sources)
        if interface == "axis":
            if not self.axis_source:
                raise RuntimeError("AXI4-Stream interface needs a streaming design")
            top, sources = f"{top}_axis", sources + [self.axis_source]
            kw.setdefault("clock_port", "aclk")
        elif interface != "native":
            raise ValueError(f"Unknown interface {interface!r}")
        out = os.path.join(self._project_dir, "efinity")
        xml = efinix.gen_project(out, top, sources, family=family, device=device, **kw)
        return efinix.run_flow(out, top, flow) if flow else xml

    def _call_cosim(self, *args):
        # Inputs are numbered per channel: arrays when streaming, scalars otherwise
        vals = [
            a for a, i in zip(args, self._arg_info) if i["is_array"] == self._is_stream
        ]
        fixed = [
            f
            for f, i in zip(self._fixed_args, self._arg_info)
            if i["is_array"] == self._is_stream
        ]
        for k, arg in enumerate(vals):
            if k < len(fixed) and fixed[k]:  # real value -> raw fixed-point bits
                arg = np.fix(np.asarray(arg, np.float64) * 2 ** fixed[k][2])
            arr = np.asarray(arg).astype(np.int64).flatten()
            arr.tofile(os.path.join(self._project_dir, f"input{k}.data"))
        res = subprocess.run(
            [self._binary_path],
            cwd=self._project_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"cosim failed:\n{res.stderr}{res.stdout}")
        m = re.search(r"CYCLES:(\d+)", res.stdout)
        self.cycles = int(m.group(1)) if m else None
        out = np.fromfile(os.path.join(self._project_dir, "output.data"), np.int64)
        # The testbench zero-extends the output port; restore the sign
        w = self._out_width
        if self._out_signed and w < 64:
            out = np.where(out >= (1 << (w - 1)), out - (1 << w), out)
        if self._fixed_out:
            out = out.astype(np.float64) / 2**self._out_frac
        if not self._is_stream:
            return float(out[0]) if self._fixed_out else int(out[0])
        return out.reshape(self._output_info["shape"]) if self._output_info else out

    def __repr__(self):
        return self.final_cpp

    # Print RAM rewrites textproto if available
    def print_textproto(self):
        if self.rewrites_textproto:
            print("=" * 60)
            print("RAM REWRITES TEXTPROTO:")
            print("=" * 60)
            print(self.rewrites_textproto)
        else:
            print("No RAM rewrites (not in memory mode)")
