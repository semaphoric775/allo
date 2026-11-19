# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""XLS[cc] C++ -> XLS IR -> vendor-neutral Verilog, plus Verilator cosim.

Tool discovery: each XLS binary is looked up in ``$XLS_BIN`` first, then PATH.
xlscc's synthesis headers (``xls_int.h`` / ac_datatypes) are passed through
``configs["include_dirs"]`` or ``$XLS_INCLUDE`` (colon separated).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

RTL_HARNESS = Path(__file__).parent.parent / "harness" / "xls_rtl"

DEFAULT_RTL_CONFIGS = {
    # XLS has no FPGA delay model; "unit" counts one op per delay unit, so
    # clock_period_ps bounds the logic depth of each pipeline stage.
    "delay_model": "unit",
    "clock_period_ps": 8,
    "pipeline_stages": None,
    "reset": "rst",
    "use_system_verilog": False,
    "include_dirs": None,
    # extra command-line flags for xlscc (list of strings)
    "xlscc_flags": None,
    # Cycles between activations of the top proc. None: aim for full
    # throughput (see throughput_first), else accept what the scheduler suggests.
    "worst_case_throughput": None,
    # The unit delay model is pessimistic for FPGA fabric, so prefer relaxing
    # clock_period_ps (up to 4x) over halving throughput when both are options.
    "throughput_first": True,
    "max_cycles": 1_000_000,
}

# Logical XLS RAM channel suffixes after ram rewrite (see _gen_textproto).
# XLS names rewritten channels "<to_name_prefix>_<suffix>", i.e. "v0__read_req".
RAM_CHANNELS = {
    "RAM_1R1W": ("read_req", "read_resp", "write_req", "write_completion"),
    "RAM_1RW": ("req", "resp", "write_completion"),
}
CTRL_PORTS = {"clk"}


def find_tool(name):
    xls_bin = os.environ.get("XLS_BIN")
    if xls_bin and os.path.isfile(os.path.join(xls_bin, name)):
        return os.path.join(xls_bin, name)
    return shutil.which(name)


def rtl_tools_available(cosim=False):
    tools = ["xlscc", "opt_main", "codegen_main"] + (["verilator"] if cosim else [])
    return all(find_tool(t) for t in tools)


def _run(cmd, cwd, stdout_path=None):
    with (
        open(stdout_path, "w", encoding="utf-8")
        if stdout_path
        else open(os.devnull, "w", encoding="utf-8")
    ) as out:
        res = subprocess.run(
            cmd, cwd=cwd, stdout=out, stderr=subprocess.PIPE, text=True, check=False
        )
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")


def _tool(name):
    path = find_tool(name)
    if path is None:
        raise RuntimeError(
            f"XLS tool '{name}' not found. Set $XLS_BIN or add it to PATH."
        )
    return path


def cpp_to_ir(cpp_path, project, has_class, include_dirs=None, extra_flags=None):
    """Run xlscc. Class-based (streaming) designs are lowered to a proc."""
    project = os.path.abspath(project)  # tools run with cwd=project
    # xlscc's synth_only headers static_assert unless __SYNTHESIS__ is defined
    cmd = [_tool("xlscc"), os.path.abspath(cpp_path), "--defines=__SYNTHESIS__"]
    dirs = include_dirs or [
        d for d in os.environ.get("XLS_INCLUDE", "").split(":") if d
    ]
    if dirs:
        cmd.append("--include_dirs=" + ",".join(dirs))
    if has_class:
        cmd += ["--block_from_class=TestBlock", "--block_pb=block.pb"]
    cmd += list(extra_flags or [])
    _run(cmd, project, os.path.join(project, "design.ir"))
    return os.path.join(project, "design.ir")


def ram_configs(ir_text, mems):
    """Build codegen --ram_configurations for each memory. codegen_main derives
    the req->resp latency constraints from these itself.

    ``mems`` is a list of (name, kind) with kind in RAM_CHANNELS.
    """
    cfgs = []
    for name, kind in mems:
        chans = [f"{name}__{s}" for s in RAM_CHANNELS[kind]]
        missing = [c for c in chans if not re.search(rf"\b{c}\b", ir_text)]
        if missing:
            raise RuntimeError(
                f"RAM '{name}': expected channels {missing} not found in optimized IR"
            )
        tag = "1R1W" if kind == "RAM_1R1W" else "1RW"
        cfgs.append(":".join([name, tag] + chans))
    return cfgs


def ir_to_verilog(ir_path, project, module_name, configs, *, rewrites=None, mems=()):
    """opt_main + codegen_main.

    Returns (verilog_path, schedule) where schedule records the
    clock_period_ps / pipeline_stages and worst_case_throughput used.
    """
    project = os.path.abspath(project)  # tools run with cwd=project
    opt_ir = os.path.join(project, "design.opt.ir")
    cmd = [_tool("opt_main"), os.path.abspath(ir_path), f"--output_path={opt_ir}"]
    if rewrites:
        cmd.append(f"--ram_rewrites_pb={os.path.abspath(rewrites)}")
    _run(cmd, project)

    is_proc = bool(re.search(r"^(top\s+)?proc\s", _read(opt_ir), re.M))
    vpath = os.path.join(project, f"{module_name}.v")
    cmd = [
        _tool("codegen_main"),
        opt_ir,
        "--generator=pipeline",
        f"--delay_model={configs['delay_model']}",
        f"--module_name={module_name}",
        f"--use_system_verilog={str(configs['use_system_verilog']).lower()}",
        f"--output_verilog_path={vpath}",
        f"--output_signature_path={os.path.join(project, module_name)}.sig.textproto",
        f"--output_schedule_path={os.path.join(project, module_name)}.schedule.textproto",
    ]
    if configs["reset"]:
        cmd += [f"--reset={configs['reset']}", "--reset_data_path=false"]
    if not is_proc:
        cmd += ["--input_valid_signal=in_vld", "--output_valid_signal=out_vld"]
    if mems:
        cmd += ["--ram_configurations=" + ",".join(ram_configs(_read(opt_ir), mems))]
    for sched in _schedule_candidates(configs):
        flags = [f"--{k}={v}" for k, v in sched.items() if v]
        try:
            _run(cmd + flags, project)
            return vpath, sched
        except RuntimeError as err:
            # Stateful procs may not sustain one activation per cycle.
            m = re.search(r"--worst_case_throughput=(\d+)", str(err))
            if not m:
                raise
            suggested = int(m.group(1))
    # Fall back to the throughput the scheduler reports as feasible.
    sched = {
        **_timing(configs, configs["clock_period_ps"]),
        "worst_case_throughput": suggested,
    }
    _run(cmd + [f"--{k}={v}" for k, v in sched.items() if v], project)
    return vpath, sched


def _timing(configs, clock_period_ps):
    if configs["pipeline_stages"]:
        return {"pipeline_stages": configs["pipeline_stages"]}
    return {"clock_period_ps": clock_period_ps}


def _schedule_candidates(configs):
    """Schedules to try in order before falling back to reduced throughput."""
    if configs["worst_case_throughput"]:
        yield {
            **_timing(configs, configs["clock_period_ps"]),
            "worst_case_throughput": configs["worst_case_throughput"],
        }
        return
    clock = configs["clock_period_ps"]
    relax = configs["throughput_first"] and not configs["pipeline_stages"]
    limit = 4 * clock if relax else clock
    while clock <= limit:
        yield _timing(configs, clock)
        clock = max(clock + 1, int(clock * 1.25))


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


PORT_RE = re.compile(r"\b(input|output)\s+wire\s+(?:\[(\d+):0\]\s+)?(\w+)")


def parse_ports(verilog, module_name):
    """Return [(dir, name, width)] of ``module_name``'s port list."""
    m = re.search(rf"module\s+{module_name}\s*\((.*?)\);", verilog, re.S)
    if not m:
        raise RuntimeError(f"Module {module_name} not found in generated Verilog")
    return [(d, n, int(w) + 1 if w else 1) for d, w, n in PORT_RE.findall(m.group(1))]


def _vec(width):
    return f"[{width - 1}:0] " if width > 1 else ""


def _ram_ports(name, kind):
    if kind == "RAM_1R1W":
        return {
            f"{name}_rd_addr": "rd_addr",
            f"{name}_rd_en": "rd_en",
            f"{name}_rd_data": "rd_data",
            f"{name}_wr_addr": "wr_addr",
            f"{name}_wr_data": "wr_data",
            f"{name}_wr_en": "wr_en",
        }
    return {
        f"{name}_addr": "addr",
        f"{name}_wr_data": "wr_data",
        f"{name}_we": "we",
        f"{name}_re": "re",
        f"{name}_rd_data": "rd_data",
    }


def gen_top(module_name, ports, mems, top_name="allo_top"):
    """Wrap the XLS module and instantiate a RAM model per external RAM.

    ``mems`` is a list of (name, kind, depth). The wrapper exposes every
    non-RAM port of the XLS module unchanged. Mask ports are left unconnected.
    """
    widths = {n: w for _, n, w in ports}
    ram_port_names = set()
    insts = []
    for name, kind, depth in mems:
        pmap = _ram_ports(name, kind)
        ram_port_names |= set(pmap)
        ram_port_names |= {
            n for _, n, _ in ports if n in (f"{name}_rd_mask", f"{name}_wr_mask")
        }
        data = f"{name}_rd_data"
        addr = f"{name}_rd_addr" if kind == "RAM_1R1W" else f"{name}_addr"
        mod = "allo_ram_1r1w" if kind == "RAM_1R1W" else "allo_ram_1rw"
        conns = ",\n    ".join(f".{v}({k})" for k, v in pmap.items())
        insts.append(
            f"  {mod} #(.WIDTH({widths[data]}), .DEPTH({depth}), .AW({widths[addr]}))"
            f" u_{name} (\n    .clk(clk),\n    {conns}\n  );"
        )
    ext = [p for p in ports if p[1] not in ram_port_names]
    wires = [f"  wire {_vec(w)}{n};" for _, n, w in ports if n in ram_port_names]
    decls = ",\n".join(f"  {d} wire {_vec(w)}{n}" for d, n, w in ext)
    conns = ",\n    ".join(f".{n}({n})" for _, n, _ in ports)
    return (
        f"// Auto-generated by Allo: top-level wrapper for {module_name}\n"
        f"module {top_name} (\n{decls}\n);\n"
        + "\n".join(wires)
        + f"\n  {module_name} u_core (\n    {conns}\n  );\n"
        + "\n".join(insts)
        + "\nendmodule\n"
    )


def gen_axis_wrapper(top_name, ports, out_words, reset="rst"):
    """AXI4-Stream view of a streaming top: every ready/valid channel ``c``
    becomes ``s_axis_<c>_*`` (inputs) or ``m_axis_<c>_*`` (outputs), reset
    becomes active-low ``aresetn`` and ``tlast`` marks the last of the
    ``out_words`` words of each result (an int, or a dict per output channel).
    Handshakes are blocked during reset:
    the XLS input buffers would otherwise accept (and then drop) data."""
    names = {n for _, n, _ in ports}
    dirs = {n: d for d, n, _ in ports}
    widths = {n: w for _, n, w in ports}
    chans = sorted(n for n in names if f"{n}_vld" in names and f"{n}_rdy" in names)
    decls = ["  input wire aclk", "  input wire aresetn"]
    conns = [".clk(aclk)"] + ([f".{reset}(~aresetn)"] if reset in names else [])
    body = []
    for c in chans:
        w = widths[c]
        if dirs[c] == "input":
            p = f"s_axis_{c}"
            decls += [
                f"  input wire {_vec(w)}{p}_tdata",
                f"  input wire {p}_tvalid",
                f"  output wire {p}_tready",
            ]
            body += [
                f"  wire {c}_rdy;",
                f"  assign {p}_tready = {c}_rdy & aresetn;",
            ]
            conns += [
                f".{c}({p}_tdata)",
                f".{c}_vld({p}_tvalid & aresetn)",
                f".{c}_rdy({c}_rdy)",
            ]
        else:
            p = f"m_axis_{c}"
            decls += [
                f"  output wire {_vec(w)}{p}_tdata",
                f"  output wire {p}_tvalid",
                f"  input wire {p}_tready",
                f"  output wire {p}_tlast",
            ]
            body += [
                f"  wire {c}_vld;",
                f"  assign {p}_tvalid = {c}_vld & aresetn;",
            ]
            conns += [
                f".{c}({p}_tdata)",
                f".{c}_vld({c}_vld)",
                f".{c}_rdy({p}_tready & aresetn)",
            ]
            words = out_words[c] if isinstance(out_words, dict) else out_words
            cw = max(1, (words - 1).bit_length())
            body += [
                f"  reg [{cw - 1}:0] {c}_cnt;",
                f"  assign {p}_tlast = {c}_cnt == {cw}'d{words - 1};",
                "  always @(posedge aclk) begin",
                f"    if (!aresetn) {c}_cnt <= 0;",
                f"    else if ({p}_tvalid && {p}_tready)",
                f"      {c}_cnt <= {p}_tlast ? 0 : {c}_cnt + 1'b1;",
                "  end",
            ]
    extra = [
        (d, n, w)
        for d, n, w in ports
        if n not in {"clk", reset}
        and n not in chans
        and not any(n in (f"{c}_vld", f"{c}_rdy") for c in chans)
    ]
    if extra:
        raise RuntimeError(f"AXI-Stream wrapper: unsupported ports {extra}")
    return (
        f"// Auto-generated by Allo: AXI4-Stream wrapper for {top_name}\n"
        f"module {top_name}_axis (\n"
        + ",\n".join(decls)
        + "\n);\n"
        + "\n".join(body)
        + f"\n  {top_name} u_top (\n    "
        + ",\n    ".join(conns)
        + "\n  );\nendmodule\n"
    )


def _cty(width):
    if width > 64:
        raise RuntimeError(f"cosim does not support ports wider than 64 bits ({width})")
    return "uint64_t" if width > 32 else "uint32_t"


def gen_tb(
    top_name, ports, in_chans, out_chan, *, n_in, n_out, stream, reset, max_cycles
):
    """Verilator C++ testbench.

    Data files hold int64 words; outputs are zero-extended port values.
    Streaming mode drives ready/valid channels from ``input<k>.data`` files and
    collects ``n_out`` words from ``out_chan`` (or one count per channel when
    ``out_chan`` is a list). Function mode applies one input vector with
    ``in_vld`` and waits for ``out_vld``. A single output goes to output.data,
    several to output<k>.data.
    """
    w = {n: wd for _, n, wd in ports}
    outs = [out_chan] if isinstance(out_chan, str) else list(out_chan)
    counts = [n_out] if isinstance(n_out, int) else list(n_out)
    files = (
        ["output.data"]
        if len(outs) == 1
        else [f"output{k}.data" for k in range(len(outs))]
    )
    lines = [
        "// Auto-generated by Allo: Verilator testbench",
        f'#include "V{top_name}.h"',
        '#include "verilated.h"',
        "#include <cstdint>\n#include <cstdio>\n#include <fstream>\n#include <vector>",
        "static std::vector<int64_t> load(const char* f, size_t n) {",
        "  std::vector<int64_t> v(n); std::ifstream s(f, std::ios::binary);",
        "  s.read(reinterpret_cast<char*>(v.data()), n * 8); return v; }",
        "int main(int argc, char** argv) {",
        "  Verilated::commandArgs(argc, argv);",
        f"  V{top_name}* m = new V{top_name};",
        "  auto tick = [&]() { m->clk = 1; m->eval(); m->clk = 0; m->eval(); };",
    ]
    for k, (c, n) in enumerate(zip(in_chans, n_in)):
        lines.append(f'  auto in{k} = load("input{k}.data", {n}); size_t i{k} = 0;')
    for k, n in enumerate(counts):
        lines.append(f"  std::vector<int64_t> out{k}; out{k}.reserve({n});")
    done = " && ".join(f"out{k}.size() >= {n}" for k, n in enumerate(counts))
    lines.append("  m->clk = 0;")
    if reset:
        lines += [
            f"  m->{reset} = 1;",
            "  for (int r = 0; r < 4; ++r) tick();",
            f"  m->{reset} = 0;",
        ]
    lines.append("  uint64_t cyc = 0;")
    if stream:
        lines.append(f"  for (; !({done}) && cyc < {max_cycles}ULL; ++cyc) {{")
        for k, (c, n) in enumerate(zip(in_chans, n_in)):
            lines.append(
                f"    m->{c}_vld = i{k} < {n}; m->{c} = ({_cty(w[c])})(i{k} < {n} ? in{k}[i{k}] : 0);"
            )
        for k, (c, n) in enumerate(zip(outs, counts)):
            lines.append(f"    m->{c}_rdy = out{k}.size() < {n};")
        lines.append("    m->eval();")
        lines += [
            f"    bool f{k} = m->{c}_vld && m->{c}_rdy;" for k, c in enumerate(in_chans)
        ]
        lines += [
            f"    if (m->{c}_vld && m->{c}_rdy) out{k}.push_back((int64_t)m->{c});"
            for k, c in enumerate(outs)
        ]
        lines.append("    tick();")
        lines += [f"    if (f{k}) ++i{k};" for k in range(len(in_chans))]
        lines.append("  }")
    else:
        for k, c in enumerate(in_chans):
            lines.append(f"  m->{c} = ({_cty(w[c])})in{k}[0];")
        lines += [
            "  m->in_vld = 1; m->eval(); tick(); m->in_vld = 0; m->eval();",
            f"  for (cyc = 1; !m->out_vld && cyc < {max_cycles}ULL; ++cyc) tick();",
            f"  if (m->out_vld) out0.push_back((int64_t)m->{outs[0]});",
        ]
    lines.append(
        f'  if (!({done})) {{ fprintf(stderr, "TIMEOUT after %llu cycles\\n", (unsigned long long)cyc); return 2; }}'
    )
    for k, f in enumerate(files):
        lines += [
            f'  {{ std::ofstream o("{f}", std::ios::binary);',
            f"    o.write(reinterpret_cast<const char*>(out{k}.data()), out{k}.size() * 8); }}",
        ]
    lines += [
        '  printf("CYCLES:%llu\\n", (unsigned long long)cyc);',
        "  m->final(); delete m; return 0;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def build_cosim(project, top_name, sources, tb_path):
    """Compile the Verilator simulation binary; returns its path."""
    project = os.path.abspath(project)
    cmd = [
        shutil.which("verilator") or "verilator",
        "--cc",
        "--exe",
        "--build",
        "-O2",
        "-Wno-fatal",
        "-Wno-lint",
        "--top-module",
        top_name,
        "-Mdir",
        "obj_dir",
        "-o",
        "sim",
        *[os.path.abspath(s) for s in sources],
        os.path.abspath(tb_path),
    ]
    _run(cmd, project)
    return os.path.join(project, "obj_dir", "sim")
