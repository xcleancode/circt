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

  void rewriteModuleSignature(FModuleOp module, InstanceGraph *graph);
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
    rewriteModuleSignature(module, instanceGraph);

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
  if (auto mem = value.getDefiningOp<MemOp>())
    return;

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

/// Update an ArrayAttribute by replacing one entry.
static ArrayAttr replaceArrayAttrElement(ArrayAttr array, size_t elem,
                                         Attribute newVal) {
  SmallVector<Attribute> old(array.begin(), array.end());
  old[elem] = newVal;
  return ArrayAttr::get(array.getContext(), old);
}

/// Construct the annotation array with a new thing appended.
static ArrayAttr appendArrayAttr(ArrayAttr array, Attribute a) {
  if (!array)
    return ArrayAttr::get(a.getContext(), ArrayRef<Attribute>{a});
  SmallVector<Attribute> old(array.begin(), array.end());
  old.push_back(a);
  return ArrayAttr::get(a.getContext(), old);
}

void IMCombCycleResolverPass::rewriteModuleSignature(FModuleOp module,
                                                     InstanceGraph *graph) {
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(module.getBodyBlock()))
    return;
  auto portInfos = module.getPortAnnotationsAttr();
  for (auto v : module.getArguments()) {
    if (v.getType().isa<FVectorType>()) {
      if (isAssumedDead(v))
        numCanPreserve++;
      else {
        // ArrayAttr
        auto newVal = appendArrayAttr(
            portInfos[v.getArgNumber()].cast<ArrayAttr>(),
            DictionaryAttr::get(
                module.getContext(),
                {NamedAttribute(StringAttr::get(module.getContext(), "class"),
                                StringAttr::get(module.getContext(),
                                                "circt.lowerAggregate"))}));
        portInfos =
            replaceArrayAttrElement(portInfos, v.getArgNumber(), newVal);
        numMustLowered++;
      }
    }
  }
  module->setAttr("portAnnotations", portInfos);
  auto node = instanceGraph->lookup(module.moduleNameAttr());
  for (auto n : node->uses()) {
    auto inst = cast<firrtl::InstanceOp>(n->getInstance());
    auto anno = inst.getPortAnnotationsAttr();
    for (auto v : module.getArguments()) {
      if (v.getType().isa<FVectorType>()) {
        if (isKnownAlive(v)) {
          auto newVal = appendArrayAttr(
              anno[v.getArgNumber()].cast<ArrayAttr>(),
              DictionaryAttr::get(
                  module.getContext(),
                  {NamedAttribute(StringAttr::get(module.getContext(),
                                                  "circt.lowerAggregate"),
                                  UnitAttr::get(module.getContext()))}));
          anno = replaceArrayAttrElement(anno, v.getArgNumber(), newVal);
        }
      }
    }
    inst.setPortAnnotationsAttr(anno);
  }

  return;
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMCombCycleResolverPass() {
  return std::make_unique<IMCombCycleResolverPass>();
}
