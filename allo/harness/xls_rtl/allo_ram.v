// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Vendor-neutral RAM models backing XLS external RAM ports (latency = 1).
// Coded in the canonical synchronous-read style so that FPGA tools
// (e.g., Efinix Efinity, Yosys, Vivado) infer block RAM.

module allo_ram_1r1w #(
    parameter WIDTH = 32,
    parameter DEPTH = 16,
    parameter AW = 4
) (
    input  wire             clk,
    input  wire [AW-1:0]    rd_addr,
    input  wire             rd_en,
    output reg  [WIDTH-1:0] rd_data,
    input  wire [AW-1:0]    wr_addr,
    input  wire [WIDTH-1:0] wr_data,
    input  wire             wr_en
);
  reg [WIDTH-1:0] mem [0:DEPTH-1];
  always @(posedge clk) begin
    if (wr_en) mem[wr_addr] <= wr_data;
    if (rd_en) rd_data <= mem[rd_addr];
  end
endmodule

module allo_ram_1rw #(
    parameter WIDTH = 32,
    parameter DEPTH = 16,
    parameter AW = 4
) (
    input  wire             clk,
    input  wire [AW-1:0]    addr,
    input  wire [WIDTH-1:0] wr_data,
    input  wire             we,
    input  wire             re,
    output reg  [WIDTH-1:0] rd_data
);
  reg [WIDTH-1:0] mem [0:DEPTH-1];
  always @(posedge clk) begin
    if (we) mem[addr] <= wr_data;
    if (re) rd_data <= mem[addr];
  end
endmodule

// Ready/valid FIFO connecting dataflow kernels (allo.stream). DEPTH must be
// a power of two >= 2. Kept in registers (Efinity syn_ramstyle); deep FIFOs
// use allo_fifo_ram instead so they can map to block RAM.
module allo_fifo #(
    parameter WIDTH = 32,
    parameter DEPTH = 2,
    parameter AW = 1
) (
    input  wire             clk,
    input  wire             rst,
    input  wire [WIDTH-1:0] in,
    input  wire             in_vld,
    output wire             in_rdy,
    output wire [WIDTH-1:0] out,
    output wire             out_vld,
    input  wire             out_rdy
);
  (* syn_ramstyle = "registers" *) reg [WIDTH-1:0] mem [0:DEPTH-1];
  reg [AW:0] wr_ptr, rd_ptr;
  wire empty = wr_ptr == rd_ptr;
  wire full = (wr_ptr[AW-1:0] == rd_ptr[AW-1:0]) && (wr_ptr[AW] != rd_ptr[AW]);
  assign in_rdy = !full;
  assign out_vld = !empty;
  assign out = mem[rd_ptr[AW-1:0]];
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr <= 0;
      rd_ptr <= 0;
    end else begin
      if (in_vld && !full) begin
        mem[wr_ptr[AW-1:0]] <= in;
        wr_ptr <= wr_ptr + 1'b1;
      end
      if (out_rdy && !empty) rd_ptr <= rd_ptr + 1'b1;
    end
  end
endmodule

// Same FIFO with the storage left to the synthesis tool (block RAM).
module allo_fifo_ram #(
    parameter WIDTH = 32,
    parameter DEPTH = 2,
    parameter AW = 1
) (
    input  wire             clk,
    input  wire             rst,
    input  wire [WIDTH-1:0] in,
    input  wire             in_vld,
    output wire             in_rdy,
    output wire [WIDTH-1:0] out,
    output wire             out_vld,
    input  wire             out_rdy
);
  reg [WIDTH-1:0] mem [0:DEPTH-1];
  reg [AW:0] wr_ptr, rd_ptr;
  wire empty = wr_ptr == rd_ptr;
  wire full = (wr_ptr[AW-1:0] == rd_ptr[AW-1:0]) && (wr_ptr[AW] != rd_ptr[AW]);
  assign in_rdy = !full;
  assign out_vld = !empty;
  assign out = mem[rd_ptr[AW-1:0]];
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr <= 0;
      rd_ptr <= 0;
    end else begin
      if (in_vld && !full) begin
        mem[wr_ptr[AW-1:0]] <= in;
        wr_ptr <= wr_ptr + 1'b1;
      end
      if (out_rdy && !empty) rd_ptr <= rd_ptr + 1'b1;
    end
  end
endmodule

// Broadcast one ready/valid stream to N consumers (a region argument read by
// several dataflow kernels). A word is retired once every consumer took it.
module allo_bcast #(
    parameter WIDTH = 32,
    parameter N = 2
) (
    input  wire             clk,
    input  wire             rst,
    input  wire [WIDTH-1:0] in,
    input  wire             in_vld,
    output wire             in_rdy,
    output wire [WIDTH-1:0] out,
    output wire [N-1:0]     out_vld,
    input  wire [N-1:0]     out_rdy
);
  reg [N-1:0] taken;
  wire [N-1:0] fire = out_vld & out_rdy;
  assign out = in;
  assign out_vld = {N{in_vld}} & ~taken;
  assign in_rdy = in_vld && &(taken | fire);
  always @(posedge clk) begin
    if (rst || in_rdy) taken <= {N{1'b0}};
    else taken <= taken | fire;
  end
endmodule

// Merge N streams of {written, data} words (a region argument written by
// several kernels, element by element in the same order): each element is
// taken from the last writer that wrote it, 0 if none did.
module allo_merge #(
    parameter WIDTH = 32,
    parameter N = 2
) (
    input  wire [N*(WIDTH+1)-1:0] in,
    input  wire [N-1:0]           in_vld,
    output wire [N-1:0]           in_rdy,
    output reg  [WIDTH-1:0]       out,
    output wire                   out_vld,
    input  wire                   out_rdy
);
  integer i;
  assign out_vld = &in_vld;
  assign in_rdy = {N{out_vld & out_rdy}};
  always @(*) begin
    out = {WIDTH{1'b0}};
    for (i = 0; i < N; i = i + 1)
      if (in[i*(WIDTH+1)+WIDTH]) out = in[i*(WIDTH+1) +: WIDTH];
  end
endmodule
