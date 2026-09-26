// ============================================================================
// test_regression_opencl_kernel_closure.cpp
// ============================================================================
// Regression: the OpenCL scan-only kernel embed must be self-contained.
//
// A consumer that wants only the scan half of this library's OpenCL kernels
// (a BIP-352 batch scanner, for instance) embeds six files --
//
//     secp256k1_field.cl  secp256k1_point.cl  secp256k1_gen_table_w8.cl
//     secp256k1_extended.cl  secp256k1_affine.cl  secp256k1_bip352.cl
//
// -- concatenates them, strips every #include line (clCreateProgramWithSource
// has no include path), and compiles the result at runtime. It never enqueues
// ecdsa_sign, schnorr_sign or ecdh.
//
// GitHub issue #415: at v4.5.0 that embed compiled. Then secp256k1_extended.cl
// grew four `#include "secp256k1_ct_*.cl"` lines and sign paths that call into
// them. Those four files are not in the embed set, so with includes stripped
// every ct_* symbol and CT* type is undefined -- and OpenCL does not dead-strip
// a function whose callees are unresolved, so the sign wrappers break the
// compile even though nothing calls them:
//
//     <kernel>:3348:5: error: unknown type name 'CTJacobianPoint'
//     <kernel>:3349:5: warning: implicit declaration of ct_generator_mul_impl
//
// This is issue #335 (Metal, SECP256K1_METAL_SCAN_ONLY) reproduced in OpenCL,
// and the remedy is the same: SECP256K1_OPENCL_SCAN_ONLY excludes the includes
// and everything that reaches them.
//
// What this module checks, on every platform, with no OpenCL device, no vendor
// compiler and no host preprocessor:
//
//   OKC-1  all six embed files resolve from any CWD
//   OKC-2  with the guard applied, the embed contains no ct_* identifier and
//          no CT* type name -- the property the consumer actually needs
//   OKC-3  WITHOUT the guard the embed does contain them, so OKC-2 cannot pass
//          vacuously if the guard were deleted
//   OKC-4  no #else sits at the top level of a SCAN_ONLY-guarded region, which
//          is the one shape the guard evaluator below does not model
//
// The evaluator is deliberately not a C preprocessor. It copies every line
// except the regions opened by a directive that mentions
// SECP256K1_OPENCL_SCAN_ONLY in its excluding form, tracking #if nesting to
// find each matching #endif. That is exactly the transformation the vendor
// compiler performs on those regions, and it needs no toolchain to be present.
// ============================================================================

#include <cstdio>
#include <cstddef>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

namespace {

constexpr const char* kKernelDir = "src/opencl/kernels/";

// The consumer's embed set, in the consumer's order (GitHub issue #415).
const char* const kEmbedFiles[] = {
    "secp256k1_field.cl",
    "secp256k1_point.cl",
    "secp256k1_gen_table_w8.cl",
    "secp256k1_extended.cl",
    "secp256k1_affine.cl",
    "secp256k1_bip352.cl",
};
constexpr std::size_t kEmbedCount = sizeof(kEmbedFiles) / sizeof(kEmbedFiles[0]);

constexpr const char* kGuardMacro = "SECP256K1_OPENCL_SCAN_ONLY";

std::vector<std::string> split_lines(const std::string& src) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (pos <= src.size()) {
        std::size_t const eol = src.find('\n', pos);
        if (eol == std::string::npos) {
            if (pos < src.size()) out.push_back(src.substr(pos));
            break;
        }
        std::string line = src.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
        pos = eol + 1;
    }
    return out;
}

// The directive word after '#', or "" when the line is not a directive.
std::string directive(const std::string& line) {
    std::size_t const h = line.find_first_not_of(" \t");
    if (h == std::string::npos || line[h] != '#') return "";
    std::size_t const b = line.find_first_not_of(" \t", h + 1);
    if (b == std::string::npos) return "";
    std::size_t const e = line.find_first_of(" \t(", b);
    return line.substr(b, (e == std::string::npos ? line.size() : e) - b);
}

bool opens_conditional(const std::string& d) {
    return d == "if" || d == "ifdef" || d == "ifndef";
}

// True for a directive that EXCLUDES its region when the guard is defined:
// "#ifndef SECP256K1_OPENCL_SCAN_ONLY", or an "#if" whose condition contains
// "!defined(SECP256K1_OPENCL_SCAN_ONLY)".
bool excludes_when_guarded(const std::string& line, const std::string& d) {
    if (line.find(kGuardMacro) == std::string::npos) return false;
    if (d == "ifndef") return true;
    if (d != "if") return false;
    return line.find("!defined(" + std::string(kGuardMacro) + ")") != std::string::npos
        || line.find("!defined (" + std::string(kGuardMacro) + ")") != std::string::npos;
}

bool is_include(const std::string& line) { return directive(line) == "include"; }

// The embed the consumer builds: six files concatenated, #include lines
// dropped. With apply_guard, regions excluded by SECP256K1_OPENCL_SCAN_ONLY
// are dropped too. stray_else counts #else directives at the top level of a
// dropped region -- a shape this evaluator does not model (OKC-4).
std::string build_embed(bool apply_guard,
                        std::vector<std::string>& unresolved,
                        int& stray_else) {
    std::string out;
    stray_else = 0;

    for (std::size_t f = 0; f < kEmbedCount; ++f) {
        std::string const src =
            audit_read_source_file((std::string(kKernelDir) + kEmbedFiles[f]).c_str());
        if (src.empty()) { unresolved.push_back(kEmbedFiles[f]); continue; }

        int skip_depth = 0;   // >0 while inside a dropped region; its nesting depth
        for (auto const& line : split_lines(src)) {
            std::string const d = directive(line);

            if (skip_depth > 0) {
                if (opens_conditional(d)) {
                    ++skip_depth;
                } else if (d == "endif") {
                    --skip_depth;
                } else if (d == "else" && skip_depth == 1) {
                    ++stray_else;
                }
                continue;                      // the whole region goes away
            }

            if (apply_guard && opens_conditional(d) && excludes_when_guarded(line, d)) {
                skip_depth = 1;
                continue;
            }
            if (is_include(line)) continue;    // the consumer strips these

            out += line;
            out += '\n';
        }
    }
    return out;
}

bool ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9') || c == '_';
}

// Whole-word occurrences of ct_<lower> identifiers and CT<Upper> type names.
// Comments count: a comment naming ct_generator_mul_impl is harmless to the
// compiler, so the scan skips // and /* */ to avoid false positives, but any
// surviving CODE reference is a hard failure.
std::vector<std::string> ct_references(const std::string& src) {
    std::vector<std::string> found;
    bool in_line_comment = false, in_block_comment = false;

    for (std::size_t i = 0; i < src.size(); ++i) {
        if (in_line_comment) { if (src[i] == '\n') in_line_comment = false; continue; }
        if (in_block_comment) {
            if (src[i] == '*' && i + 1 < src.size() && src[i + 1] == '/') {
                in_block_comment = false; ++i;
            }
            continue;
        }
        if (src[i] == '/' && i + 1 < src.size()) {
            if (src[i + 1] == '/') { in_line_comment = true; ++i; continue; }
            if (src[i + 1] == '*') { in_block_comment = true; ++i; continue; }
        }
        if (i > 0 && ident_char(src[i - 1])) continue;

        bool const lower = src.compare(i, 3, "ct_") == 0;
        bool const upper = src.compare(i, 2, "CT") == 0
                        && i + 2 < src.size() && src[i + 2] >= 'A' && src[i + 2] <= 'Z';
        if (!lower && !upper) continue;

        std::size_t j = i;
        while (j < src.size() && ident_char(src[j])) ++j;
        std::string const word = src.substr(i, j - i);
        if (word.size() > 3 || upper) {
            bool seen = false;
            for (auto const& w : found) if (w == word) { seen = true; break; }
            if (!seen) found.push_back(word);
        }
        i = j - 1;
    }
    return found;
}

// ---------------------------------------------------------------------------
// Helpers for the GH-436 static-inline guard (see the comment above
// test_regression_opencl_static_inline_link_run() below for full scope).
// ---------------------------------------------------------------------------

// True if `word` occurs in `line` as a standalone identifier -- neither
// neighbor is an identifier character. Case-sensitive, like the C/OpenCL
// keywords it is used to find.
bool contains_word(const std::string& line, const std::string& word) {
    std::size_t pos = 0;
    while ((pos = line.find(word, pos)) != std::string::npos) {
        bool const left_ok = (pos == 0) || !ident_char(line[pos - 1]);
        std::size_t const end = pos + word.size();
        bool const right_ok = (end >= line.size()) || !ident_char(line[end]);
        if (left_ok && right_ok) return true;
        pos = end;
    }
    return false;
}

// Same comment-skipping rule as ct_references() above, but blanks comment
// text instead of dropping it, so every newline (including ones inside a
// block comment) survives and split_lines() output still lines up with the
// original source's line numbers.
std::string strip_comments_keep_lines(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    bool in_line_comment = false, in_block_comment = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        char const c = src[i];
        if (in_line_comment) {
            if (c == '\n') { in_line_comment = false; out += c; }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && i + 1 < src.size() && src[i + 1] == '/') { in_block_comment = false; ++i; }
            else if (c == '\n') out += c;
            continue;
        }
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') { in_line_comment = true; ++i; continue; }
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '*') { in_block_comment = true; ++i; continue; }
        out += c;
    }
    return out;
}

// Scans `src` (comments stripped first) line by line and appends one
// "label:line: text" entry to `offenders` for each:
//   - a non-#define line with a standalone "inline" and no standalone
//     "static" (a plain inline function declaration/definition), or
//   - a "#define FORCE_INLINE ..." line whose replacement text has no
//     standalone "static".
// Returns the number of offenders appended. This is a line-oriented text
// scan, not a preprocessor: declaration specifiers split across physical
// lines are not modeled (see the caller's comment for full scope).
int scan_bad_inline(const std::string& src, const std::string& label,
                     std::vector<std::string>& offenders) {
    std::vector<std::string> const lines = split_lines(strip_comments_keep_lines(src));
    int bad = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        std::string const& line = lines[i];
        bool is_offender;
        if (directive(line) == "define") {
            is_offender = contains_word(line, "FORCE_INLINE") && !contains_word(line, "static");
        } else {
            is_offender = contains_word(line, "inline") && !contains_word(line, "static");
        }
        if (is_offender) {
            offenders.push_back(label + ":" + std::to_string(i + 1) + ": " + line);
            ++bad;
        }
    }
    return bad;
}

// Extracts and concatenates every C++ raw-string body delimited by
// R"KERNEL( ... )KERNEL" found in `cpp_src` -- the embedded OpenCL C
// kernel_parts entries in opencl_context.cpp. A C++ raw string cannot
// contain its own closing delimiter, so the first )KERNEL" after an opening
// R"KERNEL( is always the matching terminator.
std::string extract_kernel_parts(const std::string& cpp_src) {
    static const std::string kOpen = "R\"KERNEL(";
    static const std::string kClose = ")KERNEL\"";
    std::string out;
    std::size_t pos = 0;
    while (true) {
        std::size_t const start = cpp_src.find(kOpen, pos);
        if (start == std::string::npos) break;
        std::size_t const body_start = start + kOpen.size();
        std::size_t const end = cpp_src.find(kClose, body_start);
        if (end == std::string::npos) break;
        out += cpp_src.substr(body_start, end - body_start);
        out += '\n';
        pos = end + kClose.size();
    }
    return out;
}

// Enumerates *.cl files directly under src/opencl/kernels/ (no recursion),
// resolved the same CWD-independent way audit_read_source_file() resolves
// individual files: prefer the compile-time UFSECP_SOURCE_ROOT the main
// CMake build bakes in, else fall back to a path relative to this process's
// CWD. A directory that fails to resolve simply yields no entries here --
// the caller treats an empty result as the hard failure (SIL-1).
std::vector<std::string> list_cl_kernel_files() {
    std::string const dir =
#ifdef UFSECP_SOURCE_ROOT
        std::string(UFSECP_SOURCE_ROOT) + "/" + kKernelDir;
#else
        std::string(kKernelDir);
#endif
    std::vector<std::string> names;
    std::error_code ec;
    for (auto const& entry : std::filesystem::directory_iterator(dir, ec)) {
        std::error_code file_ec;
        if (!entry.is_regular_file(file_ec)) continue;
        std::string const name = entry.path().filename().string();
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".cl") == 0) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

} // namespace

int test_regression_opencl_kernel_closure_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: the OpenCL scan-only kernel embed is self-contained\n");
    std::printf("======================================================================\n\n");

    std::vector<std::string> unresolved;
    int guarded_stray_else = 0;
    std::string const guarded = build_embed(true, unresolved, guarded_stray_else);

    // OKC-1 -- the embed set exists where the consumer expects it.
    for (auto const& missing : unresolved) {
        std::printf("    UNRESOLVED: src/opencl/kernels/%s\n", missing.c_str());
    }
    CHECK(unresolved.empty(),
          "OKC-1: all six scan-only embed files resolve from any CWD");
    if (!unresolved.empty()) {
        std::printf("\n[regression_opencl_kernel_closure] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    // OKC-4 -- the evaluator models the guard shapes actually in the tree.
    CHECK(guarded_stray_else == 0,
          "OKC-4: no #else at the top level of a SECP256K1_OPENCL_SCAN_ONLY region "
          "(the guard is exclude-only; an #else arm would need a real preprocessor)");

    // OKC-2 -- the property the consumer needs. Each leaked name is printed
    // because the name is the diagnosis: it says which secp256k1_ct_*.cl file
    // the embed now depends on and does not carry.
    std::vector<std::string> const leaked = ct_references(guarded);
    std::printf("  guarded embed: %zu lines\n", split_lines(guarded).size());
    for (auto const& w : leaked) {
        std::printf("    LEAKED: %s  (defined in a secp256k1_ct_*.cl the consumer "
                    "does not embed -- this is issue #415)\n", w.c_str());
    }
    CHECK(leaked.empty(),
          "OKC-2: the -D" + std::string(kGuardMacro)
              + " embed references no ct_* symbol and no CT* type");

    // OKC-3 -- negative control: delete the guard and this must go red.
    std::vector<std::string> unresolved_plain;
    int plain_stray_else = 0;
    std::string const plain = build_embed(false, unresolved_plain, plain_stray_else);
    std::vector<std::string> const present = ct_references(plain);
    std::printf("  unguarded embed: %zu lines, %zu ct reference(s)\n",
                split_lines(plain).size(), present.size());
    for (auto const& w : present) std::printf("    %s\n", w.c_str());
    CHECK(!present.empty(),
          "OKC-3: without the guard the embed DOES reference ct_* / CT* -- if this "
          "fails, OKC-2 is passing vacuously and proves nothing");

    // The guard must remove lines, not merely exist.
    CHECK(guarded.size() < plain.size(),
          "OKC-3: the guard removes source from the embed");

    std::printf("\n[regression_opencl_kernel_closure] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

// GH-436 regression guard: every OpenCL helper must be static inline (or a
// FORCE_INLINE macro whose own expansion contains static) so AMD/ROCm (and
// any C99-inline vendor) always has an external definition even when the
// compiler chooses not to inline a large helper (field_inv_impl,
// field_sqr_impl, sha streaming bodies, bip chacha, keccak etc). Bare
// "inline" was the root cause of "undefined hidden symbol" at
// clBuildProgram time.
//
// What this checks: every *.cl file in src/opencl/kernels/ (enumerated from
// disk, not a hardcoded list -- 26 files today) plus the OpenCL C embedded
// as kernel_parts raw strings in src/opencl/src/opencl_context.cpp, scanned
// (comments stripped) for a non-static "inline" function declaration or
// definition, and every "#define FORCE_INLINE ..." whose replacement text
// has no "static". A missing/unreadable input is a hard failure, not a
// skip (SIL-1/SIL-2 below).
// What this does NOT check: it is a source-text scan, not a preprocessor or
// a vendor build -- it does not expand macros, does not model declaration
// specifiers split across physical lines, and proves nothing about whether
// a given vendor's OpenCL compiler actually accepts the kernel (that is the
// GPU CI matrix's job). src/bch/opencl and src/ltc/opencl are separate
// kernel trees and are out of scope for this guard.
int test_regression_opencl_static_inline_link_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: OpenCL static-inline link hygiene (GH-436)\n");
    std::printf("======================================================================\n\n");

    std::vector<std::string> offenders;
    std::vector<std::string> unresolved;

    std::vector<std::string> const kernel_files = list_cl_kernel_files();
    std::printf("  kernel files found: %zu\n", kernel_files.size());
    for (auto const& name : kernel_files) {
        std::string const rel = std::string(kKernelDir) + name;
        std::string const src = audit_read_source_file(rel.c_str());
        if (src.empty()) { unresolved.push_back(rel); continue; }
        scan_bad_inline(src, rel, offenders);
    }
    for (auto const& missing : unresolved) {
        std::printf("    UNRESOLVED: %s\n", missing.c_str());
    }
    CHECK(!kernel_files.empty(),
          "SIL-1: src/opencl/kernels/ resolves from this CWD and contains at least one .cl file");
    CHECK(unresolved.empty(), "SIL-1: every enumerated kernel file is readable");
    if (kernel_files.empty() || !unresolved.empty()) {
        std::printf("\n[regression_opencl_static_inline_link] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    std::string const ctx_path = "src/opencl/src/opencl_context.cpp";
    std::string const ctx_src = audit_read_source_file(ctx_path.c_str());
    CHECK(!ctx_src.empty(), "SIL-2: src/opencl/src/opencl_context.cpp resolves from this CWD");
    if (ctx_src.empty()) {
        std::printf("\n[regression_opencl_static_inline_link] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    std::string const kernel_parts_src = extract_kernel_parts(ctx_src);
    CHECK(!kernel_parts_src.empty(),
          "SIL-2: opencl_context.cpp contains at least one R\"KERNEL( ... )KERNEL\" kernel_parts entry");
    if (!kernel_parts_src.empty()) {
        scan_bad_inline(kernel_parts_src, ctx_path + " (kernel_parts)", offenders);
    }

    for (auto const& off : offenders) {
        std::printf("    OFFENDER: %s\n", off.c_str());
    }
    CHECK(offenders.empty(),
          "SIL-3: no non-static \"inline\" function decl/def and no non-static "
          "FORCE_INLINE define in any scanned OpenCL source");

    // SIL-4 -- negative control: the scanner itself must catch the bad
    // shapes and leave the good ones alone, or SIL-3 could be passing
    // vacuously (same idea as OKC-3 above).
    std::vector<std::string> neg_bad;
    std::vector<std::string> neg_good;
    scan_bad_inline("inline void f(void) {}\n", "<neg>", neg_bad);
    scan_bad_inline("#define FORCE_INLINE inline\n", "<neg>", neg_bad);
    scan_bad_inline("static inline void f(void) {}\n", "<pos>", neg_good);
    scan_bad_inline("#define FORCE_INLINE static inline\n", "<pos>", neg_good);
    scan_bad_inline("FORCE_INLINE void f(void) {}\n", "<pos>", neg_good);
    scan_bad_inline("// inline void f(void) {}\n", "<pos>", neg_good);
    CHECK(neg_bad.size() == 2,
          "SIL-4: the scanner flags a bare `inline void f()` and a bare "
          "`#define FORCE_INLINE inline` -- if this fails, SIL-3 proves nothing");
    CHECK(neg_good.empty(),
          "SIL-4: the scanner does not flag static inline, FORCE_INLINE use, "
          "a static FORCE_INLINE define, or a commented-out inline");

    std::printf("\n[regression_opencl_static_inline_link] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() {
    int const closure_rc = test_regression_opencl_kernel_closure_run();
    int const static_inline_rc = test_regression_opencl_static_inline_link_run();
    return (closure_rc != 0 || static_inline_rc != 0) ? 1 : 0;
}
#endif
