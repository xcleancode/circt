// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{body contains non-pure operation}}
arc.define @Foo(%arg0: i1) {
  // expected-note @+1 {{first non-pure operation here:}}
  arc.state @Bar() clock %arg0 lat 1 : () -> ()
  arc.output
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo() {
  // expected-error @+1 {{'arc.state' op with non-zero latency requires a clock}}
  arc.state @Bar() lat 1 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%clock: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have a clock}}
  arc.state @Bar() clock %clock lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

hw.module @Foo(%enable: i1) {
  // expected-error @+1 {{'arc.state' op with zero latency cannot have an enable}}
  arc.state @Bar() enable %enable lat 0 : () -> ()
}
arc.define @Bar() {
  arc.output
}

// -----

// expected-error @+1 {{body contains non-pure operation}}
arc.define @SupportRecurisveMemoryEffects(%arg0: i1, %arg1: i1) {
  // expected-note @+1 {{first non-pure operation here:}}
  scf.if %arg0 {
    arc.state @Bar() clock %arg1 lat 1 : () -> ()
  }
  arc.output
}
arc.define @Bar() {
  arc.output
}
