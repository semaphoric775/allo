/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITXLSHLS_H
#define ALLO_TRANSLATION_EMITXLSHLS_H

#include "allo/Translation/EmitVivadoHLS.h"

namespace mlir {
namespace allo {
namespace hls {

/// XLS HLS ModuleEmitter (inherits from VhlsModuleEmitter)
class XlsModuleEmitter : public VhlsModuleEmitter {
public:
  using operand_range = Operation::operand_range;
  explicit XlsModuleEmitter(AlloEmitterState &state)
      : VhlsModuleEmitter(state) {}

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module) override;

  /// Override methods that need XLS-specific behavior.
  void emitFunctionDirectives(func::FuncOp func,
                              ArrayRef<Value> portList) override;
  void emitArrayDecl(Value array, bool isFunc = false,
                     std::string name = "") override;
  void emitLoopDirectives(Operation *op) override;
  void emitStreamConstruct(allo::StreamConstructOp op) override;
  void emitArrayDirectives(Value memref) override;
  /// Bit operations as shift-and-mask on ac_int (bounds may be runtime).
  void emitGetBit(allo::GetIntBitOp op) override;
  void emitSetBit(allo::SetIntBitOp op) override;
  void emitGetSlice(allo::GetIntSliceOp op) override;
  void emitSetSlice(allo::SetIntSliceOp op) override;
  /// XlsInt has no implicit narrowing to native ints: wrap integer casts.
  void emitCastOperand(Value operand, Value result) override;
  void emitFunction(func::FuncOp func) override;

  /// Affine statement emitters.
  void emitAffineFor(affine::AffineForOp op) override;

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op) override;

  /// Memref-related statement emitters.
  void emitLoad(memref::LoadOp op) override;
  void emitStore(memref::StoreOp op) override;
  void emitAffineLoad(affine::AffineLoadOp op) override;
  void emitAffineStore(affine::AffineStoreOp op) override;
  void emitAffineStoreImpl(affine::AffineStoreOp op);
  void emitStoreImpl(memref::StoreOp op);

protected:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "") override;

  /// Helper to emit flattened index for multi-dimensional array access.
  void emitFlattenedIndex(MemRefType memrefType,
                          Operation::operand_range indices);

  /// Helper method to get XLS-specific type names.
  SmallString<16> getTypeName(Type valType);
  SmallString<16> getTypeName(Value val);

  /// Partitioned arrays in memory mode. A block/cyclic partitioned array is
  /// split into one __xls_memory per bank ("<name>_b<k>"); a completely
  /// partitioned one becomes a class-level register array.
  /// `operandNames` name the dims then symbols of the access `map`.
  /// Returns the C++ expression reading the element, or "" if `memref` is not
  /// partitioned (or not in memory mode).
  std::string partitionedReadExpr(Value memref, AffineMap map,
                                  ArrayRef<std::string> operandNames);
  /// Emits the (indented) store statement(s); returns false if `memref` is
  /// not partitioned (or not in memory mode).
  bool emitPartitionedStore(Value memref, AffineMap map,
                            ArrayRef<std::string> operandNames,
                            StringRef valueStr);
  /// Emit declaration comments parsed by the Python wrapper. Returns false if
  /// the array is not partitioned.
  bool emitPartitionedDecl(Value memref, StringRef name);
  /// C++ names of index operands (integer constants are printed inline).
  SmallVector<std::string, 4> getOperandNames(ValueRange values);
  /// C++ lvalue/rvalue for element `map(operands)` of a (non-stream) array.
  std::string accessExpr(Value memref, AffineMap map,
                         ArrayRef<std::string> operandNames);
  /// Memory mode: XLS cannot see dependences through external RAMs, so a
  /// read-modify-write of one element across loop iterations (e.g. the
  /// accumulator of a reduction) would read stale data. Such accesses are
  /// kept in a register for the duration of the loop instead.
  DenseMap<Operation *, std::string> accumulatorOf;
  /// Dataflow outputs with several writers carry a per-element "written" bit
  /// (array arg -> mask array name), see attribute "xls_masked_outputs".
  DenseMap<Value, std::string> outputMask;
  void emitMaskSet(Value memref, AffineMap map,
                   ArrayRef<std::string> operandNames);

  /// Track array arguments that should become local arrays populated from
  /// channels.
  bool inArrayFunction = false;
  DenseMap<Value, std::string> arrayArgToLocalName;
  DenseMap<Value, std::string> arrayArgToChannelName;
  Value returnArray = nullptr;
  std::string outputChannelName = "out";
};

} // namespace hls

LogicalResult emitXlsHLS(ModuleOp module, llvm::raw_ostream &os,
                         bool useMemory = false);
void registerEmitXlsHLSTranslation();

} // namespace allo
} // namespace mlir

#endif // ALLO_TRANSLATION_EMITXLSHLS_H
