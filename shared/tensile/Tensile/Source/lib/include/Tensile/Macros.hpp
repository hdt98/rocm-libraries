/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

// TENSILE_API controls C++ symbol visibility for Tensile types and free
// functions whose declarations appear in this directory's public headers.
//
// tensile-host is built as a STATIC library and statically composed into
// consumer shared libraries -- librocblas.so today -- typically via
// $<LINK_LIBRARY:WHOLE_ARCHIVE,...> so every TU is pulled in. The exported
// ABI of those consumer DSOs is their own C surface (e.g. rocblas_*);
// Tensile's class and template definitions are deliberately NOT part of
// that ABI.
//
// CMake's `CXX_VISIBILITY_PRESET hidden` on `tensile-host` only governs
// the visibility of symbols emitted from Tensile's own translation units.
// It does NOT govern what consumer translation units emit when they
// `#include` these headers. Inline methods, implicit template
// instantiations, and the vtable / typeinfo for any class used as a
// complete type in a consumer TU will be emitted into the consumer's
// object files under the CONSUMER's per-TU `-fvisibility=` setting.
//
// When two consumer DSOs both export the same Tensile-internal C++
// mangled name with incompatible class layouts, ELF flat-namespace
// interposition causes one DSO's calls to resolve into the other DSO's
// definition and crash. This is the same failure mode that was observed
// in March 2026 between libhipblaslt.so and libhipsparselt.so sharing a
// divergent TensileLite ContractionProblemGemm; the hazard applies any
// time librocblas.so coexists in a process with another DSO that also
// embeds a Tensile.
//
// To prevent this at the source, TENSILE_API is defined as a class- and
// function-level hidden-visibility attribute. Class-level visibility
// attributes travel through the header into every consuming TU and
// OVERRIDE the consumer's per-TU `-fvisibility=` fallback for the marked
// type -- including its vtable, typeinfo, inline method bodies, and
// implicit template instantiations.
//
// On compilers without the GCC visibility attribute (notably MSVC), this
// expands to nothing and visibility falls back to the platform default;
// on those platforms the relevant ABI hazard does not exist in the same
// form.

#if defined(__GNUC__) || defined(__clang__)
#define TENSILE_API __attribute__((visibility("hidden")))
#else
#define TENSILE_API
#endif

// TENSILE_HIDDEN_BEGIN / TENSILE_HIDDEN_END wrap a region of declarations
// (typically a `namespace Tensile { ... }` block in a public header) and
// apply hidden visibility to every declaration inside, regardless of
// whether the declaration carries an explicit attribute. This is
// belt-and-braces for the per-symbol TENSILE_API marker above: it ensures
// that types added in the future without explicit annotation -- and the
// implicit weak symbols every class with virtual functions emits (vtable,
// typeinfo, typeinfo name) -- also receive hidden visibility in consumer
// translation units.
//
// Use as:
//
//     TENSILE_HIDDEN_BEGIN
//     namespace Tensile { /* declarations */ }
//     TENSILE_HIDDEN_END
//
// On compilers without GCC visibility pragmas, both macros expand to
// nothing.

#if defined(__GNUC__) || defined(__clang__)
#define TENSILE_HIDDEN_BEGIN _Pragma("GCC visibility push(hidden)")
#define TENSILE_HIDDEN_END   _Pragma("GCC visibility pop")
#else
#define TENSILE_HIDDEN_BEGIN
#define TENSILE_HIDDEN_END
#endif
