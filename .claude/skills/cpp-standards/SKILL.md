---
name: cpp-standards
description: Continuum C++ coding standards and clean-code checklist. Use when writing, refactoring, or reviewing any C++ in src/, include/, bindings/, or tests/cpp — covers RAII, ownership, const-correctness, type safety, Doxygen, and naming.
---

# C++ Coding Standards & Clean Code

You are an expert C++ software engineer specializing in modern C++ (C++17/20/23).
Follow these strict guidelines when writing, refactoring, or reviewing code.

## Core Principles

- **RAII (Resource Acquisition Is Initialization):** Manage all resources
  (memory, files, locks) via objects; never use raw `new` or `delete`.
- **Immutability by Default:** Use `const` or `constexpr` for any value that
  does not change. Practice "West const" (e.g. `const int x = 5;`).
- **Smart Pointers:** Use `std::unique_ptr` for exclusive ownership and
  `std::shared_ptr` for shared ownership. Use raw pointers only for non-owning
  observation.
- **Value Semantics:** Prefer passing objects by value or `const&`, returning
  via value (relying on RVO/move semantics), and avoiding out-parameters.
- **Type Safety:** Prefer strong typing (`enum class`) over raw integers or
  C-style enums. Use `gsl::span` or standard containers instead of raw arrays.

## Documentation & Style

- **Doxygen:** Document all public classes, structs, functions, and public
  fields using standard Doxygen tags (`@brief`, `@param`, `@return`).
- **Naming Conventions:**
  - Classes/Structs/Enums: `PascalCase`
  - Free functions and file-local helpers: `PascalCase` (project convention)
  - Methods on bound/public classes: `snake_case` (project convention)
  - Local variables and parameters: `snake_case`
  - Constants: `kPascalCase`; macros: `UPPER_CASE`
- **Formatting:** Comply strictly with the project `.clang-format` and address
  `.clang-tidy` findings.

## Project specifics

- Standard is C++20 (`CMAKE_CXX_STANDARD 20`; raised from 17 because current libtorch headers require it). Do not use later-standard
  features without raising the standard in `CMakeLists.txt` first.
- Warnings are errors-adjacent: `-Wall -Wextra -Wpedantic` are on. Keep new
  code warning-clean.
- The engine core in `src/` must not depend on pybind11 or Python headers;
  that boundary lives only in `bindings/pybind/`.
- Cross-backend tensor conversions must be explicit. No silent conversions.
