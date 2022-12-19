//===- IMDeadCodeElim.cpp - Intermodule Dead Code Elimination ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-imdeadcodeelim"

using namespace circt;
using namespace firrtl;

bool checkIfValueUsedAsSink(Value v) {
  SmallVector<Value> worklist = {v};
  while (!worklist.empty()) {
    auto e = worklist.pop_back_val();
    for (auto *user : e.getUsers()) {
      if (isa<FConnectLike>(user) && user->getOperand(0) == e)
        return true;
      if (isa<SubindexOp, SubaccessOp, SubfieldOp>(user))
        worklist.push_back(user->getResult(0));
    }
  }
  return false;
}

bool checkIfValueUsedAsSource(Value v) {
  SmallVector<Value> worklist = {v};
  while (!worklist.empty()) {
    auto e = worklist.pop_back_val();
    for (auto *user : e.getUsers()) {
      if (isa<FConnectLike>(user) && user->getOperand(0) == e)
        continue;
      else if (isa<SubindexOp, SubaccessOp, SubfieldOp>(user)) {
        worklist.push_back(user->getResult(0));
      } else {
        return true;
      }
    }
  }
  return false;
}

bool noCycle(Value wire) {
  StrictConnectOp writer = getSingleConnectUserOf(wire);
  if (!writer)
    return false;
  for (auto *user : wire.getUsers()) {
    if (user != writer) {
      if (user->isBeforeInBlock(writer))
        return false;
    }
  }
  return true;
}

bool usedAsWire(Value v) {
  return checkIfValueUsedAsSink(v) && checkIfValueUsedAsSource(v) &&
         !noCycle(v);
}

namespace {
struct IMCombCycleResolverPass
    : public IMCombCycleResolverBase<IMCombCycleResolverPass> {
  void runOnOperation() override;

  void rewriteModuleSignature(FModuleOp module);
  void rewriteModuleBody(FModuleOp module);

  void markAlive(Value value) {
    //  If the value is already in `liveSet`, skip it.
    if (liveSet.insert(value).second)
      worklist.push_back(value);
  }

  /// Return true if the value is known alive.
  bool isKnownAlive(Value value) const {
    assert(value && "null should not be used");
    return liveSet.count(value);
  }

  /// Return true if the value is assumed dead.
  bool isAssumedDead(Value value) const { return !isKnownAlive(value); }
  bool isAssumedDead(Operation *op) const {
    return llvm::none_of(op->getResults(),
                         [&](Value value) { return isKnownAlive(value); });
  }

  /// Return true if the block is alive.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  void visitUser(Operation *op);
  void visitValue(Value value);
  void visitConnect(FConnectLike connect);
  void visitSubelement(Operation *op);
  void markBlockExecutable(Block *block);
  void markDeclaration(Operation *op);
  void markInstanceOp(InstanceOp instanceOp);

private:
  /// The set of blocks that are known to execute, or are intrinsically alive.
  DenseSet<Block *> executableBlocks;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<BlockArgument, llvm::TinyPtrVector<Value>>
      resultPortToInstanceResultMapping;
  InstanceGraph *instanceGraph;

  /// A worklist of values whose liveness recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<Value, 64> worklist;
  llvm::DenseSet<Value> liveSet;
};
} // namespace

void IMCombCycleResolverPass::markDeclaration(Operation *op) {
  if (!noCycle(op->getResult(0)))
    for (auto result : op->getResults())
      markAlive(result);
}

void IMCombCycleResolverPass::visitUser(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");
  if (auto connectOp = dyn_cast<FConnectLike>(op))
    return visitConnect(connectOp);
  if (isa<SubfieldOp, SubindexOp, SubaccessOp>(op))
    return visitSubelement(op);
}

void IMCombCycleResolverPass::markInstanceOp(InstanceOp instance) {
  // Get the module being referenced.
  Operation *op = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any inputs and inouts are
  // alive.
  if (!isa<FModuleOp>(op)) {
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(op);
  markBlockExecutable(fModule.getBodyBlock());

  // Ok, it is a normal internal module reference so populate
  // resultPortToInstanceResultMapping.
  for (auto resultNo : llvm::seq(0u, instance.getNumResults())) {
    auto instancePortVal = instance.getResult(resultNo);
    if (usedAsWire(instancePortVal))
      markAlive(instancePortVal);

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);
  }
}

void IMCombCycleResolverPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  // Mark ports with don't touch as alive.
  for (auto blockArg : block->getArguments())
    if (usedAsWire(blockArg))
      markAlive(blockArg);

  for (auto &op : *block) {
    if (isa<WireOp>(&op))
      markDeclaration(&op);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);

    // TODO: Handle attach etc.
  }
}

void IMCombCycleResolverPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();

  for (auto module : circuit.getBodyBlock()->getOps<FModuleOp>()) {
    // Mark the ports of public modules as alive.
    markBlockExecutable(module.getBodyBlock());
  }

  // If a value changed liveness then propagate liveness through its users and
  // definition.
  while (!worklist.empty())
    visitValue(worklist.pop_back_val());

  // Rewrite module signatures.
  for (auto module : circuit.getBodyBlock()->getOps<FModuleOp>())
    rewriteModuleSignature(module);

  // Rewrite module bodies parallelly.
  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBodyBlock()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModuleBody(op); });
}

void IMCombCycleResolverPass::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  // Propagate liveness through users.
  for (Operation *user : value.getUsers())
    visitUser(user);

  // Requiring an input port propagates the liveness to each instance.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    auto module = cast<FModuleOp>(blockArg.getParentBlock()->getParentOp());

    for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
      markAlive(userOfResultPort);
    return;
  }

  // Marking an instance port as alive propagates to the corresponding port of
  // the module.
  if (auto instance = value.getDefiningOp<InstanceOp>()) {
    auto instanceResult = value.cast<mlir::OpResult>();
    // Update the src, when it's an instance op.
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));

    // Propagate liveness only when a port is output.
    if (!module)
      return;

    BlockArgument modulePortVal =
        module.getArgument(instanceResult.getResultNumber());
    return markAlive(modulePortVal);
  }

  // If a port of a memory is alive, all other ports are.
  if (auto mem = value.getDefiningOp<MemOp>()) {
    return;
  }

  // If op is defined by an operation, mark its operands as alive.
  if (auto op = value.getDefiningOp())
    for (auto operand : op->getOperands())
      markAlive(operand);
}

void IMCombCycleResolverPass::visitConnect(FConnectLike connect) {
  // If the dest is alive, mark the source value as alive.
  if (isa_and_nonnull<RegResetOp, RegOp>(
          circt::firrtl::getFieldRefFromValue(connect.getDest())
              .getValue()
              .getDefiningOp()))
    return;
  if (isKnownAlive(connect.getDest()))
    markAlive(connect.getSrc());
}

void IMCombCycleResolverPass::visitSubelement(Operation *op) {
  if (isKnownAlive(op->getOperand(0)))
    markAlive(op->getResult(0));
}

void IMCombCycleResolverPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBodyBlock();
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(body))
    return;
}

namespace {
/// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  /// This is the underlying ground type of the field.
  FIRRTLBaseType type;
  /// The index in the parent type
  size_t index;
  /// The fieldID
  unsigned fieldID;
  /// This is a suffix to add to the field name to make it unique.
  SmallString<16> suffix;
  /// This indicates whether the field was flipped to be an output.
  bool isOutput;

  FlatBundleFieldEntry(const FIRRTLBaseType &type, size_t index,
                       unsigned fieldID, StringRef suffix, bool isOutput)
      : type(type), index(index), fieldID(fieldID), suffix(suffix),
        isOutput(isOutput) {}

  void dump() const {
    llvm::errs() << "FBFE{" << type << " index<" << index << "> fieldID<"
                 << fieldID << "> suffix<" << suffix << "> isOutput<"
                 << isOutput << ">}\n";
  }
};
} // namespace

static bool peelType(Type type, SmallVectorImpl<FlatBundleFieldEntry> &fields) {
  if (auto refType = type.dyn_cast<RefType>())
    type = refType.getType();
  return TypeSwitch<Type, bool>(type)
      .Case<BundleType>([&](auto bundle) {
        SmallString<16> tmpSuffix;
        // Otherwise, we have a bundle type.  Break it down.
        for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
          auto elt = bundle.getElement(i);
          // Construct the suffix to pass down.
          tmpSuffix.resize(0);
          tmpSuffix.push_back('_');
          tmpSuffix.append(elt.name.getValue());
          fields.emplace_back(elt.type, i, bundle.getFieldID(i), tmpSuffix,
                              elt.isFlip);
        }
        return true;
      })
      .Case<FVectorType>([&](auto vector) {
        // Increment the field ID to point to the first element.
        for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
          fields.emplace_back(vector.getElementType(), i, vector.getFieldID(i),
                              "_" + std::to_string(i), false);
        }
        return true;
      })
      .Default([](auto op) { return false; });
}

void IMCombCycleResolverPass::rewriteModuleSignature(FModuleOp module) {
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(module.getBodyBlock()))
    return;

  for (auto v : module.getArguments()) {
    if (v.getType().isa<FVectorType>()) {
      if (isAssumedDead(v))
        numCanPreserve++;
      else {
        numMustLowered++;
      }
    }
  }

  return;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMCombCycleResolverPass() {
  return std::make_unique<IMCombCycleResolverPass>();
}
