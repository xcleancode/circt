//===- UBSanitizer.cpp - SFC Compatible Pass ----------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass makes a number of updates to the circuit that are required to match
// the behavior of the Scala FIRRTL Compiler (SFC).  This pass removes invalid
// values from the circuit.  This is a combination of the Scala FIRRTL
// Compiler's RemoveRests pass and RemoveValidIf.  This is done to remove two
// "interpretations" of invalid.  Namely: (1) registers that are initialized to
// an invalid value (module scoped and looking through wires and connects only)
// are converted to an unitialized register and (2) invalid values are converted
// to zero (after rule 1 is applied).  Additionally, this pass checks and
// disallows async reset registers that are not driven with a constant when
// looking through wires, connects, and nodes.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-resets"

using namespace circt;
using namespace firrtl;

struct UBSanitizerPass : public UBSanitizerBase<UBSanitizerPass> {
  void runOnOperation() override;
};
static bool derivedByOtherValue(Value value, Value reg) {
  if (value == reg)
    return false;
  auto op = value.getDefiningOp();
  // Ports are ok for now.
  if (!op)
    return true;
  if (isa<WireOp, RegResetOp, RegOp>(op))
    return true;

  if (auto it = dyn_cast<MuxPrimOp>(op))
    return derivedByOtherValue(it.getHigh(), reg) ||
           derivedByOtherValue(it.getLow(), reg);
  if (llvm::all_of(op->getOperands(),
                   [&](Value v) { return derivedByOtherValue(v, reg); })) {
    return true;
  }
  return false;
}
void UBSanitizerPass::runOnOperation() {
  // Assume that this pass is performed after ExpandWHens

  getOperation().walk([&](StrictConnectOp connect) {
    auto dest = connect.getDest();
    auto reg = dest.getDefiningOp<RegOp>();
    if (!reg)
      return;

    auto src = connect.getSrc();
    if (!derivedByOtherValue(src, reg)) {
      auto diag = reg.emitError() << "uninilitiazed";
      diag.attachNote(dest.getLoc()) << "dest value";
      signalPassFailure();
    }
  });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createUBSanitizerPass() {
  return std::make_unique<UBSanitizerPass>();
}
