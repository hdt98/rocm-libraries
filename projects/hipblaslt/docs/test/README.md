# `hipblaslt` Tests

## 1. Test Overview

hipBLASLt and TensileLite have two largely independent test stacks that reflect the layered
architecture of the project:

```
┌─────────────────────────────────────────────────────────────────┐
│  hipBLASLt API Layer                                            │
│  C++ GTest (hipblaslt-test)                                     │
│  Tests: matmul, auxiliary ops, extended ops, matrix transform   │
│  Framework: Google Test + YAML-driven test data                 │
├─────────────────────────────────────────────────────────────────┤
│  TensileLite Host Library                                       │
│  C++ GTest (tensilelite-tests)                                  │
│  Tests: kernel selection, predicate matching, data types        │
│  Framework: Google Test, no GPU required                        │
├─────────────────────────────────────────────────────────────────┤
│  TensileLite Kernel Generation (Python)                         │
│  pytest (Tensile/Tests/)                                        │
│  Tests: codegen logic, scheduling, parameter validation         │
│  Framework: pytest + pytest-xdist, GPU required for common/     │
├─────────────────────────────────────────────────────────────────┤
│  rocISA Instruction Abstraction Layer                           │
│  pytest (rocisa/test/)                                          │
│  Tests: instruction generation, code containers, labels         │
│  Framework: pytest, no GPU required                             │
└─────────────────────────────────────────────────────────────────┘
```

## 2. `hipblaslt-test`

## 3. `tensilelite-test`

## 4. Tensile Code Generation Tests

## 5. `rocIsa`
